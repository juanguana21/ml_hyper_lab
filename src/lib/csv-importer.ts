/**
 * ML HyperLab — CSV Importer
 * Parses CSV files client-side using PapaParse.
 * Handles large files with streaming, auto-detects types, and prepares
 * data for ML training.
 */

import Papa from 'papaparse';

export interface ImportedDataset {
  name: string;
  fileName: string;
  X: number[][];
  y: number[];
  feature_names: string[];
  target_names: string[];
  rawColumns: string[];
  targetColumnName: string;
  nSamples: number;
  nFeatures: number;
  nClasses: number;
  taskType: 'classification' | 'regression';
  preview: string[][]; // first 5 rows for display
  columnTypes: ColumnTypeInfo[];
}

export interface ColumnTypeInfo {
  name: string;
  type: 'numeric' | 'categorical' | 'id' | 'date' | 'empty';
  uniqueValues: number;
  sampleValues: string[];
  nullCount: number;
}

export interface ParseProgress {
  stage: 'reading' | 'parsing' | 'analyzing' | 'preparing' | 'ready' | 'error';
  progress: number; // 0-100
  message: string;
  rowsProcessed?: number;
  totalRows?: number;
}

export function parseCSVFile(
  file: File,
  onProgress?: (progress: ParseProgress) => void
): Promise<{
  columns: string[];
  rows: string[][];
  columnTypes: ColumnTypeInfo[];
  preview: string[][];
}> {
  return new Promise((resolve, reject) => {
    onProgress?.({ stage: 'reading', progress: 0, message: 'Leyendo archivo...' });

    const rows: string[][] = [];
    let columns: string[] = [];
    let isFirstChunk = true;
    let totalRows = 0;

    Papa.parse(file, {
      header: false,
      dynamicTyping: false,
      skipEmptyLines: 'greedy',
      worker: file.size > 5 * 1024 * 1024, // Use worker for files > 5MB
      chunkSize: 1024 * 1024, // 1MB chunks
      chunk(results: Papa.ParseResult<string[]>) {
        if (isFirstChunk && results.data.length > 0) {
          columns = results.data[0].map((c) => String(c).trim());
          isFirstChunk = false;
          results.data = results.data.slice(1);
        }

        for (const row of results.data) {
          if (row.length === columns.length) {
            rows.push(row.map((v) => String(v).trim()));
          }
        }

        totalRows += results.data.length;
        const estimatedProgress = Math.min(90, Math.round((totalRows / (file.size / 50)) * 100));
        onProgress?.({
          stage: 'parsing',
          progress: estimatedProgress,
          message: `Procesando... ${totalRows.toLocaleString()} filas leídas`,
          rowsProcessed: totalRows,
        });
      },
      complete() {
        onProgress?.({ stage: 'analyzing', progress: 92, message: 'Analizando columnas...' });

        const columnTypes = analyzeColumns(columns, rows);
        const preview = rows.slice(0, 6);

        onProgress?.({ stage: 'ready', progress: 100, message: 'Archivo cargado' });

        resolve({ columns, rows, columnTypes, preview });
      },
      error(err: Error) {
        onProgress?.({ stage: 'error', progress: 0, message: `Error: ${err.message}` });
        reject(err);
      },
    });
  });
}

function analyzeColumns(columns: string[], rows: string[][]): ColumnTypeInfo[] {
  const nRows = rows.length;
  return columns.map((col, colIdx) => {
    const values = rows.map((r) => r[colIdx]);
    const nonEmpty = values.filter((v) => v !== '' && v !== 'NA' && v !== 'null' && v !== 'NaN' && v !== 'N/A' && v !== '-');
    const nullCount = nRows - nonEmpty.length;
    const uniqueValues = new Set(nonEmpty).size;

    // Sample values (up to 5)
    const sampleValues = [...new Set(nonEmpty)].slice(0, 5);

    // Detect type
    let type: ColumnTypeInfo['type'] = 'categorical';

    if (nonEmpty.length === 0) {
      type = 'empty';
    } else {
      // Check if mostly numeric
      const numericCount = nonEmpty.filter((v) => !isNaN(Number(v)) && v.trim() !== '').length;
      const numericRatio = numericCount / nonEmpty.length;

      if (numericRatio > 0.9) {
        type = 'numeric';
      } else if (uniqueValues === nRows && nRows > 10) {
        // Likely an ID column
        type = 'id';
      } else if (
        nonEmpty.some((v) =>
          /\d{4}[-\/]\d{1,2}[-\/]\d{1,2}/.test(v) ||
          /\d{1,2}[-\/]\d{1,2}[-\/]\d{4}/.test(v)
        )
      ) {
        // Contains date-like patterns
        if (numericRatio < 0.5) type = 'date';
      } else {
        type = 'categorical';
      }
    }

    return {
      name: col,
      type,
      uniqueValues,
      sampleValues,
      nullCount,
    };
  });
}

/**
 * Detect if a target column represents regression (continuous) or classification (discrete)
 */
export function detectTaskType(
  targetColumnName: string,
  columnTypes: ColumnTypeInfo[],
  columns: string[],
  rows: string[][]
): 'classification' | 'regression' {
  const targetIdx = columns.indexOf(targetColumnName);
  if (targetIdx === -1) return 'classification';

  const ct = columnTypes[targetIdx];
  if (ct.type !== 'numeric') return 'classification'; // categorical target = classification

  const values = rows.map((r) => r[targetIdx])
    .filter((v) => v !== '' && v !== 'NA' && v !== 'null' && v !== 'NaN' && v !== 'N/A' && v !== '-');
  const numericValues = values.map(Number).filter((v) => !isNaN(v));

  if (numericValues.length === 0) return 'classification';

  const uniqueValues = new Set(numericValues);
  const uniqueRatio = uniqueValues.size / numericValues.length;

  // If many unique values relative to total, it's likely regression
  // Heuristic: >20 unique values AND >10% unique ratio = regression
  // Also check if values are not just integers in a small range
  const allIntegers = numericValues.every((v) => Number.isInteger(v));
  const smallRange = uniqueValues.size <= 20;

  if (uniqueValues.size > 20 && uniqueRatio > 0.1) return 'regression';
  if (!allIntegers && uniqueValues.size > 5) return 'regression';
  if (smallRange && allIntegers) return 'classification';
  if (uniqueValues.size > 10) return 'regression';

  return 'classification';
}

export function prepareDataset(
  columns: string[],
  rows: string[][],
  columnTypes: ColumnTypeInfo[],
  targetColumnName: string,
  datasetName: string,
  fileName: string,
  maxRows: number = 10000,
  overrideTaskType?: 'classification' | 'regression'
): ImportedDataset {
  const targetColIdx = columns.indexOf(targetColumnName);

  // Determine feature columns (exclude target, IDs, dates, empty columns)
  const featureColIndices: number[] = [];
  const featureNames: string[] = [];
  const categoricalMappings: Map<string, Map<string, number>> = new Map();

  for (let i = 0; i < columns.length; i++) {
    if (i === targetColIdx) continue;
    const ct = columnTypes[i];
    if (ct.type === 'id' || ct.type === 'date' || ct.type === 'empty') continue;
    featureColIndices.push(i);
    featureNames.push(columns[i]);
    if (ct.type === 'categorical') {
      // Build mapping for categorical features
      const uniqueVals = [...new Set(rows.map((r) => r[i]))].filter(
        (v) => v !== '' && v !== 'NA' && v !== 'null' && v !== 'NaN' && v !== 'N/A'
      );
      const mapping = new Map<string, number>();
      uniqueVals.forEach((v, idx) => mapping.set(v, idx));
      categoricalMappings.set(columns[i], mapping);
    }
  }

  // Limit rows
  const useRows = rows.slice(0, maxRows);

  // Build feature matrix X
  const X: number[][] = [];
  const y: number[] = [];

  // Build target encoding
  const targetValues = useRows.map((r) => r[targetColIdx]);
  const uniqueTargetValues = [...new Set(targetValues)].sort();
  const targetMapping = new Map<string, number>();
  uniqueTargetValues.forEach((v, idx) => targetMapping.set(v, idx));

  for (const row of useRows) {
    const features: number[] = [];
    let valid = true;

    for (let fi = 0; fi < featureColIndices.length; fi++) {
      const colIdx = featureColIndices[fi];
      const colName = columns[colIdx];
      const rawVal = row[colIdx];

      if (
        rawVal === '' || rawVal === 'NA' || rawVal === 'null' ||
        rawVal === 'NaN' || rawVal === 'N/A' || rawVal === '-'
      ) {
        // Use 0 for missing values (simple imputation)
        features.push(0);
        continue;
      }

      const ct = columnTypes[colIdx];
      if (ct.type === 'numeric') {
        const num = Number(rawVal);
        if (isNaN(num)) {
          features.push(0);
        } else {
          features.push(num);
        }
      } else if (ct.type === 'categorical') {
        const mapping = categoricalMappings.get(colName);
        if (mapping) {
          features.push(mapping.get(rawVal) ?? 0);
        } else {
          features.push(0);
        }
      } else {
        features.push(0);
      }
    }

    if (!valid) continue;

    X.push(features);
    y.push(targetMapping.get(row[targetColIdx]) ?? 0);
  }

  const autoDetectedTaskType = detectTaskType(targetColumnName, columnTypes, columns, rows);
  const taskType = overrideTaskType || autoDetectedTaskType;
  const preview = rows.slice(0, 6);

  let nClasses: number;
  let targetNames: string[];

  if (taskType === 'regression') {
    // For regression, keep raw numeric y values and don't encode
    // If user overrode to regression but target has categorical values,
    // we still try to parse as numeric (label-encoded if needed)
    nClasses = 0; // Not applicable for regression
    targetNames = ['valor'];
    // Rebuild y with raw numeric values for regression
    const useRows2 = rows.slice(0, maxRows);
    const regressionY: number[] = [];
    const regressionX: number[][] = [];
    for (let ri = 0; ri < useRows2.length; ri++) {
      const row = useRows2[ri];
      const rawTarget = row[targetColIdx];
      const numTarget = Number(rawTarget);
      if (isNaN(numTarget) || rawTarget === '' || rawTarget === 'NA' || rawTarget === 'null' || rawTarget === 'NaN' || rawTarget === 'N/A' || rawTarget === '-') continue;
      const features: number[] = [];
      for (let fi = 0; fi < featureColIndices.length; fi++) {
        const colIdx = featureColIndices[fi];
        const colName = columns[colIdx];
        const rawVal = row[colIdx];
        if (rawVal === '' || rawVal === 'NA' || rawVal === 'null' || rawVal === 'NaN' || rawVal === 'N/A' || rawVal === '-') {
          features.push(0);
          continue;
        }
        const ct2 = columnTypes[colIdx];
        if (ct2.type === 'numeric') {
          features.push(isNaN(Number(rawVal)) ? 0 : Number(rawVal));
        } else if (ct2.type === 'categorical') {
          const mapping = categoricalMappings.get(colName);
          features.push(mapping ? (mapping.get(rawVal) ?? 0) : 0);
        } else {
          features.push(0);
        }
      }
      regressionX.push(features);
      regressionY.push(numTarget);
    }
    // Replace X and y with regression versions
    X.length = 0;
    for (const row of regressionX) X.push(row);
    y.length = 0;
    for (const val of regressionY) y.push(val);

    return {
      name: datasetName,
      fileName,
      X: regressionX,
      y: regressionY,
      feature_names: featureNames,
      target_names: targetNames,
      rawColumns: columns,
      targetColumnName,
      nSamples: regressionX.length,
      nFeatures: featureNames.length,
      nClasses,
      taskType,
      preview,
      columnTypes,
    };
  }

  nClasses = uniqueTargetValues.length;
  targetNames = uniqueTargetValues.map((v) =>
    v.length > 20 ? v.substring(0, 18) + '...' : v
  );

  return {
    name: datasetName,
    fileName,
    X,
    y,
    feature_names: featureNames,
    target_names: targetNames,
    rawColumns: columns,
    targetColumnName,
    nSamples: X.length,
    nFeatures: featureNames.length,
    nClasses,
    taskType,
    preview,
    columnTypes,
  };
}

/**
 * Suggest which column might be the target based on heuristics:
 * 1. If there's a column named "target", "label", "class", "y", "output", "diagnosis" etc.
 * 2. Otherwise, the last categorical column with reasonable cardinality (2-20 classes)
 * 3. Fallback: last column
 */
export function suggestTargetColumn(columns: string[], columnTypes: ColumnTypeInfo[]): string {
  const targetNames = ['target', 'label', 'class', 'y', 'output', 'diagnosis', 'species', 'survived', 'churn', 'outcome', 'category', 'type', 'respuesta', 'etiqueta', 'clase', 'categoria', 'resultado', 'diagnostico'];

  // Check for exact name match (case-insensitive)
  for (const col of columns) {
    const colLower = col.toLowerCase().replace(/[_\s-]/g, '');
    if (targetNames.some((t) => colLower.includes(t))) {
      return col;
    }
  }

  // Find last categorical column with 2-20 unique values
  for (let i = columns.length - 1; i >= 0; i--) {
    const ct = columnTypes[i];
    if (ct.type === 'categorical' && ct.uniqueValues >= 2 && ct.uniqueValues <= 20) {
      return columns[i];
    }
  }

  // Find any column with 2-20 unique values
  for (let i = columns.length - 1; i >= 0; i--) {
    const ct = columnTypes[i];
    if (ct.uniqueValues >= 2 && ct.uniqueValues <= 20 && ct.type !== 'id' && ct.type !== 'date' && ct.type !== 'empty') {
      return columns[i];
    }
  }

  // Fallback: last column
  return columns[columns.length - 1];
}
