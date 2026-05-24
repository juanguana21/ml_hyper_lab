'use client';

import React, { useState, useCallback, useRef, useEffect } from 'react';
import {
  ALGORITHMS,
  DATASET_PRESETS,
  REGRESSION_ALGORITHMS,
  REGRESSION_DATASET_PRESETS,
  AlgorithmConfig,
  HyperparamConfig,
} from '@/lib/ml-config';
import { trainModel, computeDecisionBoundary, trainModelWithData, computeDecisionBoundaryWithData, trainRegressionModel, trainRegressionModelWithData, computeRegressionPlotData, computeRegressionPlotDataWithData, RegressionTrainResult, RegressionPlotData } from '@/lib/ml-algorithms';
import { parseCSVFile, prepareDataset, suggestTargetColumn, detectTaskType, ImportedDataset, ParseProgress, ColumnTypeInfo } from '@/lib/csv-importer';
import { WORKSHOPS, Workshop, WorkshopStep } from '@/lib/workshops';
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Slider } from '@/components/ui/slider';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from '@/components/ui/tooltip';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Separator } from '@/components/ui/separator';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Checkbox } from '@/components/ui/checkbox';
import { Progress } from '@/components/ui/progress';
import {
  Play,
  RotateCcw,
  Info,
  ChevronRight,
  Brain,
  BarChart3,
  Target,
  TrendingUp,
  AlertTriangle,
  CheckCircle2,
  XCircle,
  Loader2,
  BookOpen,
  Lightbulb,
  Zap,
  Eye,
  GitCompareArrows,
  GraduationCap,
  ChevronDown,
  Trophy,
  Clock,
  HelpCircle,
  ArrowRight,
  Flame,
  Upload,
  FileSpreadsheet,
  X,
  Table,
  Columns,
  Database,
} from 'lucide-react';

// ============================================================
// Types
// ============================================================
interface Metrics {
  accuracy: number;
  precision: number;
  recall: number;
  f1: number;
}

interface RegressionMetrics {
  mse: number;
  rmse: number;
  mae: number;
  r2: number;
}

interface FeatureImportance {
  feature: string;
  importance: number;
  index: number;
}

interface TrainResult {
  success: boolean;
  metrics?: Metrics;
  confusion_matrix?: number[][];
  feature_importance?: FeatureImportance[];
  tree_text?: string | null;
  n_train?: number;
  n_test?: number;
  target_names?: string[];
  feature_names?: string[];
  error?: string;
}

interface RegressionTrainResultLocal {
  success: boolean;
  metrics?: RegressionMetrics;
  feature_importance?: FeatureImportance[];
  tree_text?: string | null;
  n_train?: number;
  n_test?: number;
  target_names?: string[];
  feature_names?: string[];
  y_test?: number[];
  y_pred?: number[];
  error?: string;
}

interface RegressionPlotDataLocal {
  points_x: number[];
  points_y: number[];
  points_true: number[];
  points_pred: number[];
  feature_names: string[];
  regression_line_x: number[];
  regression_line_y: number[];
}

interface BoundaryData {
  grid_x: number[][];
  grid_y: number[][];
  grid_z: number[][];
  points_x: number[];
  points_y: number[];
  points_labels: number[];
  feature_names: string[];
  n_classes: number;
  target_names: string[];
}

interface ComparisonResult {
  algorithmId: string;
  algorithmName: string;
  color: string;
  metrics: Metrics;
  confusion_matrix: number[][];
  boundaryData: BoundaryData | null;
  featureImportance: FeatureImportance[];
  n_train: number;
  n_test: number;
}

interface RegressionComparisonResult {
  algorithmId: string;
  algorithmName: string;
  color: string;
  metrics: RegressionMetrics;
  featureImportance: FeatureImportance[];
  n_train: number;
  n_test: number;
}

// ============================================================
// Helper: Initial hyperparameters
// ============================================================
function getInitialHyperparams(algo: AlgorithmConfig): Record<string, number | string | null> {
  const params: Record<string, number | string | null> = {};
  algo.hyperparams.forEach((hp) => {
    params[hp.key] = hp.defaultValue;
  });
  return params;
}

function prepareCleanParams(
  hyperparams: Record<string, number | string | null>,
  algoId: string
): Record<string, unknown> {
  const cleanParams: Record<string, unknown> = {};
  Object.entries(hyperparams).forEach(([key, val]) => {
    if (
      key === 'max_depth' &&
      (val === 0 || val === null) &&
      (algoId === 'decision_tree' || algoId === 'random_forest')
    ) {
      cleanParams[key] = null;
    } else {
      cleanParams[key] = val;
    }
  });
  return cleanParams;
}

// ============================================================
// Component: HyperparameterControlWrapper
// ============================================================
function HyperparameterControlWrapper({
  config,
  value,
  onChange,
  disabled,
  algoId,
}: {
  config: HyperparamConfig;
  value: number | string | null;
  onChange: (key: string, value: number | string | null) => void;
  disabled: boolean;
  algoId: string;
}) {
  const impactColor: Record<string, string> = {
    alto: 'bg-red-100 text-red-700 border-red-200',
    medio: 'bg-yellow-100 text-yellow-700 border-yellow-200',
    bajo: 'bg-green-100 text-green-700 border-green-200',
  };

  const impactIcon: Record<string, React.ReactNode> = {
    alto: <Zap className="w-3 h-3" />,
    medio: <TrendingUp className="w-3 h-3" />,
    bajo: <Target className="w-3 h-3" />,
  };

  const isMaxDepthUnlimited =
    config.key === 'max_depth' &&
    (algoId === 'decision_tree' || algoId === 'random_forest') &&
    (value === 0 || value === null || value === undefined);

  const displayValue = isMaxDepthUnlimited ? '∞' : value;

  const sliderValue =
    config.key === 'max_depth' &&
    (algoId === 'decision_tree' || algoId === 'random_forest') &&
    (value === null || value === undefined)
      ? 0
      : (value as number);

  const handleSliderChange = (v: number[]) => {
    if (
      config.key === 'max_depth' &&
      (algoId === 'decision_tree' || algoId === 'random_forest') &&
      v[0] === 0
    ) {
      onChange(config.key, null);
    } else {
      onChange(config.key, v[0]);
    }
  };

  return (
    <div className="space-y-1.5 py-2 px-3 rounded-lg border border-border/50 bg-card/50 hover:bg-card/80 transition-colors">
      <div className="flex items-center justify-between gap-2">
        <div className="flex items-center gap-2 flex-1 min-w-0">
          <label className="text-sm font-medium text-foreground truncate flex items-center gap-1.5">
            <span>{config.label}</span>
            <code className="text-[10px] font-mono bg-slate-100 text-slate-600 px-1.5 py-0.5 rounded border border-slate-200 shrink-0">
              {config.pythonParam}
            </code>
          </label>
          <TooltipProvider delayDuration={200}>
            <Tooltip>
              <TooltipTrigger asChild>
                <button
                  type="button"
                  className="h-5 w-5 inline-flex items-center justify-center rounded-md hover:bg-accent shrink-0"
                >
                  <Info className="h-3.5 w-3.5 text-muted-foreground" />
                </button>
              </TooltipTrigger>
              <TooltipContent side="right" className="max-w-sm p-3">
                <p className="text-sm font-medium mb-1">{config.label}</p>
                <p className="text-xs text-muted-foreground mb-2">{config.explanation}</p>
                <div className="flex items-start gap-1.5 bg-muted/50 rounded p-2 mb-2">
                  <Lightbulb className="h-3.5 w-3.5 text-yellow-500 shrink-0 mt-0.5" />
                  <p className="text-xs">{config.tip}</p>
                </div>
                <div className="bg-slate-800 text-green-400 rounded p-2 text-xs font-mono">
                  <span className="text-slate-500"># Python ({config.pythonLib})</span>
                  <br />
                  <span className="text-blue-300">model</span> = <span className="text-yellow-300">{config.pythonLib.split('.').pop()}</span>(<span className="text-orange-300">{config.pythonParam}</span>=<span className="text-green-300">{value}</span>)
                </div>
              </TooltipContent>
            </Tooltip>
          </TooltipProvider>
        </div>
        <Badge
          variant="outline"
          className={`text-[10px] px-1.5 py-0 shrink-0 ${impactColor[config.impact] || ''}`}
        >
          {impactIcon[config.impact]}
          <span className="ml-1">Impacto {config.impact}</span>
        </Badge>
      </div>

      <div className="flex items-center gap-3">
        {config.type === 'slider' && (
          <>
            <Slider
              min={config.min}
              max={config.max}
              step={config.step}
              value={[sliderValue ?? 0]}
              onValueChange={handleSliderChange}
              disabled={disabled}
              className="flex-1"
            />
            <span className="text-sm font-mono font-semibold min-w-[3rem] text-right text-foreground">
              {typeof displayValue === 'number'
                ? Number.isInteger(displayValue)
                  ? displayValue
                  : displayValue.toFixed(3)
                : displayValue}
            </span>
          </>
        )}
        {config.type === 'select' && (
          <Select
            value={String(value ?? '')}
            onValueChange={(v) => onChange(config.key, v)}
            disabled={disabled}
          >
            <SelectTrigger className="flex-1">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {config.options?.map((opt) => (
                <SelectItem key={String(opt.value)} value={String(opt.value)}>
                  {opt.label}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        )}
      </div>
    </div>
  );
}

// ============================================================
// Component: ConfusionMatrixHeatmap (SVG-based)
// ============================================================
function ConfusionMatrixHeatmap({
  matrix,
  targetNames,
}: {
  matrix: number[][];
  targetNames: string[];
}) {
  if (!matrix || matrix.length === 0) return null;

  const n = matrix.length;
  const cellSize = Math.min(60, 240 / n);
  const labelWidth = 80;
  const totalWidth = labelWidth + n * cellSize + 40;
  const totalHeight = labelWidth + n * cellSize + 40;

  const maxVal = Math.max(...matrix.flat());
  const getColor = (val: number) => {
    const intensity = maxVal > 0 ? val / maxVal : 0;
    const r = Math.round(255 * (1 - intensity * 0.7));
    const g = Math.round(255 * (1 - intensity * 0.3));
    const b = Math.round(255 * (1 - intensity * 0.1));
    return `rgb(${r}, ${g}, ${b})`;
  };

  const shortNames = targetNames.map((name) =>
    name.length > 12 ? name.substring(0, 10) + '…' : name
  );

  return (
    <div className="flex flex-col items-center">
      <p className="text-sm font-semibold mb-2 text-foreground">Matriz de Confusión</p>
      <svg width={totalWidth} height={totalHeight} className="overflow-visible">
        <text
          x={20}
          y={labelWidth / 2 + (n * cellSize) / 2}
          textAnchor="middle"
          transform={`rotate(-90, 20, ${labelWidth / 2 + (n * cellSize) / 2})`}
          className="text-xs fill-muted-foreground"
          fontSize={11}
        >
          Etiqueta Real
        </text>
        <text
          x={labelWidth + (n * cellSize) / 2}
          y={totalHeight - 5}
          textAnchor="middle"
          className="text-xs fill-muted-foreground"
          fontSize={11}
        >
          Predicción
        </text>
        {matrix.map((row, i) =>
          row.map((val, j) => (
            <g key={`${i}-${j}`}>
              <rect
                x={labelWidth + j * cellSize}
                y={labelWidth + i * cellSize}
                width={cellSize}
                height={cellSize}
                fill={getColor(val)}
                stroke="white"
                strokeWidth={2}
                rx={4}
              />
              <text
                x={labelWidth + j * cellSize + cellSize / 2}
                y={labelWidth + i * cellSize + cellSize / 2 + 5}
                textAnchor="middle"
                className="text-xs font-semibold"
                fill={val > maxVal * 0.5 ? 'white' : '#1a1a1a'}
                fontSize={13}
              >
                {val}
              </text>
            </g>
          ))
        )}
        {shortNames.map((name, i) => (
          <text
            key={`row-${i}`}
            x={labelWidth - 5}
            y={labelWidth + i * cellSize + cellSize / 2 + 4}
            textAnchor="end"
            className="text-xs fill-foreground"
            fontSize={10}
          >
            {name}
          </text>
        ))}
        {shortNames.map((name, j) => (
          <text
            key={`col-${j}`}
            x={labelWidth + j * cellSize + cellSize / 2}
            y={labelWidth - 5}
            textAnchor="middle"
            className="text-xs fill-foreground"
            fontSize={10}
          >
            {name}
          </text>
        ))}
      </svg>
    </div>
  );
}

// ============================================================
// Component: DecisionBoundaryCanvas
// ============================================================
function DecisionBoundaryCanvas({ data }: { data: BoundaryData | null }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    if (!data || !canvasRef.current) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const width = 400;
    const height = 400;
    canvas.width = width;
    canvas.height = height;

    const gridX = data.grid_x;
    const gridY = data.grid_y;
    const gridZ = data.grid_z;
    const rows = gridZ.length;
    const cols = gridZ[0]?.length || 0;

    if (rows === 0 || cols === 0) return;

    const xMin = Math.min(...gridX[0]);
    const xMax = Math.max(...gridX[0]);
    const yMin = Math.min(...gridY.map((r) => r[0]));
    const yMax = Math.max(...gridY.map((r) => r[0]));

    const colorPalette = ['#10b981', '#f59e0b', '#8b5cf6', '#ef4444', '#06b6d4', '#ec4899'];
    const colorPaletteLight = [
      '#d1fae5', '#fef3c7', '#ede9fe', '#fee2e2', '#cffafe', '#fce7f3',
    ];

    const cellW = width / cols;
    const cellH = height / rows;

    for (let i = 0; i < rows; i++) {
      for (let j = 0; j < cols; j++) {
        const cls = Math.round(gridZ[i][j]);
        ctx.fillStyle = colorPaletteLight[cls % colorPaletteLight.length];
        ctx.fillRect(j * cellW, i * cellH, cellW + 1, cellH + 1);
      }
    }

    if (data.points_x && data.points_y) {
      data.points_x.forEach((px, idx) => {
        const py = data.points_y[idx];
        const label = data.points_labels[idx];

        const xRange = xMax - xMin || 1;
        const yRange = yMax - yMin || 1;
        const cx = ((px - xMin) / xRange) * width;
        const cy = ((py - yMin) / yRange) * height;

        ctx.beginPath();
        ctx.arc(cx, cy, 4, 0, 2 * Math.PI);
        ctx.fillStyle = colorPalette[label % colorPalette.length];
        ctx.fill();
        ctx.strokeStyle = 'white';
        ctx.lineWidth = 1.5;
        ctx.stroke();
      });
    }

    ctx.fillStyle = '#374151';
    ctx.font = '12px system-ui';
    ctx.textAlign = 'center';
    ctx.fillText(data.feature_names[0] || 'Feature 1', width / 2, height - 8);
    ctx.save();
    ctx.translate(14, height / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText(data.feature_names[1] || 'Feature 2', 0, 0);
    ctx.restore();

    const legendY = 10;
    if (data.target_names) {
      data.target_names.forEach((name, i) => {
        const lx = width - 120;
        const ly = legendY + i * 20;
        ctx.fillStyle = colorPalette[i % colorPalette.length];
        ctx.beginPath();
        ctx.arc(lx, ly + 6, 5, 0, 2 * Math.PI);
        ctx.fill();
        ctx.fillStyle = '#374151';
        ctx.font = '11px system-ui';
        ctx.textAlign = 'left';
        const displayName = name.length > 15 ? name.substring(0, 13) + '…' : name;
        ctx.fillText(displayName, lx + 12, ly + 10);
      });
    }
  }, [data]);

  if (!data) {
    return (
      <div className="flex items-center justify-center h-[400px] bg-muted/30 rounded-lg border border-dashed">
        <div className="text-center text-muted-foreground">
          <Eye className="h-10 w-10 mx-auto mb-2 opacity-40" />
          <p className="text-sm">Selecciona un dataset y entrena el modelo</p>
          <p className="text-xs">para ver la frontera de decisión</p>
        </div>
      </div>
    );
  }

  return (
    <div className="flex flex-col items-center">
      <p className="text-sm font-semibold mb-2 text-foreground">Frontera de Decisión (2D)</p>
      <canvas
        ref={canvasRef}
        className="rounded-lg border bg-white"
        style={{ width: 400, height: 400 }}
      />
    </div>
  );
}

// ============================================================
// Component: MiniDecisionBoundaryCanvas (200x200 for comparison)
// ============================================================
function MiniDecisionBoundaryCanvas({
  data,
  width = 200,
  height = 200,
}: {
  data: BoundaryData | null;
  width?: number;
  height?: number;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    if (!data || !canvasRef.current) return;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    canvas.width = width;
    canvas.height = height;

    const gridX = data.grid_x;
    const gridY = data.grid_y;
    const gridZ = data.grid_z;
    const rows = gridZ.length;
    const cols = gridZ[0]?.length || 0;
    if (rows === 0 || cols === 0) return;

    const xMin = Math.min(...gridX[0]);
    const xMax = Math.max(...gridX[0]);
    const yMin = Math.min(...gridY.map((r: number[]) => r[0]));
    const yMax = Math.max(...gridY.map((r: number[]) => r[0]));

    const colorPalette = ['#10b981', '#f59e0b', '#8b5cf6', '#ef4444', '#06b6d4', '#ec4899'];
    const colorPaletteLight = [
      '#d1fae5', '#fef3c7', '#ede9fe', '#fee2e2', '#cffafe', '#fce7f3',
    ];

    const cellW = width / cols;
    const cellH = height / rows;
    for (let i = 0; i < rows; i++) {
      for (let j = 0; j < cols; j++) {
        const cls = Math.round(gridZ[i][j]);
        ctx.fillStyle = colorPaletteLight[cls % colorPaletteLight.length];
        ctx.fillRect(j * cellW, i * cellH, cellW + 1, cellH + 1);
      }
    }

    if (data.points_x && data.points_y) {
      data.points_x.forEach((px, idx) => {
        const py = data.points_y[idx];
        const label = data.points_labels[idx];
        const xRange = xMax - xMin || 1;
        const yRange = yMax - yMin || 1;
        const cx = ((px - xMin) / xRange) * width;
        const cy = ((py - yMin) / yRange) * height;
        ctx.beginPath();
        ctx.arc(cx, cy, 2, 0, 2 * Math.PI);
        ctx.fillStyle = colorPalette[label % colorPalette.length];
        ctx.fill();
      });
    }
  }, [data, width, height]);

  if (!data) return null;

  return (
    <canvas
      ref={canvasRef}
      className="rounded-lg border bg-white"
      style={{ width, height }}
    />
  );
}

// ============================================================
// Component: FeatureImportanceChart (SVG bar chart)
// ============================================================
function FeatureImportanceChart({ data, algorithmId }: { data: FeatureImportance[]; algorithmId: string }) {
  if (algorithmId === 'knn') {
    return (
      <div className="flex items-center justify-center h-[200px] bg-muted/30 rounded-lg border border-dashed">
        <div className="text-center">
          <AlertTriangle className="h-8 w-8 mx-auto mb-2 text-yellow-500" />
          <p className="text-sm text-muted-foreground">
            KNN no proporciona importancia de features
          </p>
          <p className="text-xs text-muted-foreground mt-1">
            KNN es un modelo basado en distancia, no tiene coeficientes o importancias
          </p>
        </div>
      </div>
    );
  }

  if (!data || data.length === 0) {
    return (
      <div className="flex items-center justify-center h-[200px] bg-muted/30 rounded-lg border border-dashed">
        <p className="text-sm text-muted-foreground">Sin datos de importancia disponibles</p>
      </div>
    );
  }

  const maxImp = Math.max(...data.map((d) => d.importance));
  const barHeight = 24;
  const chartWidth = 280;
  const labelWidth = 140;
  const totalWidth = labelWidth + chartWidth + 60;
  const totalHeight = data.length * (barHeight + 8) + 30;

  return (
    <div className="flex flex-col items-center">
      <p className="text-sm font-semibold mb-2 text-foreground">Importancia de Features</p>
      <svg width={totalWidth} height={totalHeight}>
        {data.map((item, i) => {
          const barW = maxImp > 0 ? (item.importance / maxImp) * chartWidth : 0;
          const y = i * (barHeight + 8) + 5;
          return (
            <g key={item.feature}>
              <text
                x={labelWidth - 5}
                y={y + barHeight / 2 + 4}
                textAnchor="end"
                className="text-xs fill-foreground"
                fontSize={10}
              >
                {item.feature.length > 20
                  ? item.feature.substring(0, 18) + '…'
                  : item.feature}
              </text>
              <rect
                x={labelWidth}
                y={y}
                width={barW}
                height={barHeight}
                fill="#10b981"
                rx={4}
                opacity={0.8}
              />
              <text
                x={labelWidth + barW + 5}
                y={y + barHeight / 2 + 4}
                className="text-xs fill-muted-foreground"
                fontSize={10}
              >
                {item.importance.toFixed(4)}
              </text>
            </g>
          );
        })}
      </svg>
    </div>
  );
}

// ============================================================
// Component: RegressionScatterCanvas (actual vs predicted)
// ============================================================
function RegressionScatterCanvas({ data }: { data: RegressionPlotDataLocal | null }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    if (!data || !canvasRef.current) return;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const width = 400;
    const height = 400;
    canvas.width = width;
    canvas.height = height;

    const yTrue = data.points_true;
    const yPred = data.points_pred;
    if (!yTrue.length || !yPred.length) return;

    const allVals = [...yTrue, ...yPred];
    const minVal = Math.min(...allVals) - 1;
    const maxVal = Math.max(...allVals) + 1;
    const range = maxVal - minVal || 1;

    const padding = 50;
    const plotW = width - 2 * padding;
    const plotH = height - 2 * padding;

    // Background
    ctx.fillStyle = '#fafafa';
    ctx.fillRect(0, 0, width, height);

    // Axes
    ctx.strokeStyle = '#d1d5db';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(padding, padding);
    ctx.lineTo(padding, height - padding);
    ctx.lineTo(width - padding, height - padding);
    ctx.stroke();

    // Grid lines
    ctx.strokeStyle = '#e5e7eb';
    ctx.lineWidth = 0.5;
    for (let i = 0; i <= 5; i++) {
      const x = padding + (i / 5) * plotW;
      const y = height - padding - (i / 5) * plotH;
      ctx.beginPath();
      ctx.moveTo(x, padding);
      ctx.lineTo(x, height - padding);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(padding, y);
      ctx.lineTo(width - padding, y);
      ctx.stroke();

      // Labels
      ctx.fillStyle = '#6b7280';
      ctx.font = '10px system-ui';
      ctx.textAlign = 'center';
      ctx.fillText((minVal + (i / 5) * range).toFixed(1), x, height - padding + 15);
      ctx.textAlign = 'right';
      ctx.fillText((minVal + (i / 5) * range).toFixed(1), padding - 8, y + 3);
    }

    // Perfect prediction line
    ctx.strokeStyle = '#94a3b8';
    ctx.lineWidth = 1.5;
    ctx.setLineDash([6, 4]);
    ctx.beginPath();
    ctx.moveTo(padding + ((minVal - minVal) / range) * plotW, height - padding - ((minVal - minVal) / range) * plotH);
    ctx.lineTo(padding + ((maxVal - minVal) / range) * plotW, height - padding - ((maxVal - minVal) / range) * plotH);
    ctx.stroke();
    ctx.setLineDash([]);

    // Scatter points
    for (let i = 0; i < yTrue.length; i++) {
      const cx = padding + ((yTrue[i] - minVal) / range) * plotW;
      const cy = height - padding - ((yPred[i] - minVal) / range) * plotH;
      ctx.beginPath();
      ctx.arc(cx, cy, 3, 0, 2 * Math.PI);
      ctx.fillStyle = '#10b981';
      ctx.globalAlpha = 0.6;
      ctx.fill();
      ctx.globalAlpha = 1;
    }

    // Axis labels
    ctx.fillStyle = '#374151';
    ctx.font = '12px system-ui';
    ctx.textAlign = 'center';
    ctx.fillText('Valor Real', width / 2, height - 8);
    ctx.save();
    ctx.translate(14, height / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText('Valor Predicho', 0, 0);
    ctx.restore();

    // Legend
    ctx.fillStyle = '#94a3b8';
    ctx.font = '10px system-ui';
    ctx.textAlign = 'left';
    ctx.fillText('--- Predicción perfecta', width - 140, 20);
  }, [data]);

  if (!data) {
    return (
      <div className="flex items-center justify-center h-[400px] bg-muted/30 rounded-lg border border-dashed">
        <div className="text-center text-muted-foreground">
          <Eye className="h-10 w-10 mx-auto mb-2 opacity-40" />
          <p className="text-sm">Entrena un modelo de regresión</p>
          <p className="text-xs">para ver el gráfico de dispersión</p>
        </div>
      </div>
    );
  }

  return (
    <div className="flex flex-col items-center">
      <p className="text-sm font-semibold mb-2 text-foreground">Real vs Predicho</p>
      <canvas ref={canvasRef} className="rounded-lg border bg-white" style={{ width: 400, height: 400 }} />
    </div>
  );
}

// ============================================================
// Component: ResidualPlotCanvas
// ============================================================
function ResidualPlotCanvas({ yTest, yPred }: { yTest: number[]; yPred: number[] }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    if (!yTest.length || !yPred.length || !canvasRef.current) return;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const width = 400;
    const height = 300;
    canvas.width = width;
    canvas.height = height;

    const residuals = yPred.map((p, i) => p - yTest[i]);
    const maxRes = Math.max(Math.abs(Math.min(...residuals)), Math.abs(Math.max(...residuals)), 1) * 1.1;
    const minPred = Math.min(...yPred) - 1;
    const maxPred = Math.max(...yPred) + 1;
    const predRange = maxPred - minPred || 1;

    const padding = 50;
    const plotW = width - 2 * padding;
    const plotH = height - 2 * padding;

    ctx.fillStyle = '#fafafa';
    ctx.fillRect(0, 0, width, height);

    // Axes
    ctx.strokeStyle = '#d1d5db';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(padding, padding);
    ctx.lineTo(padding, height - padding);
    ctx.lineTo(width - padding, height - padding);
    ctx.stroke();

    // Zero line
    const zeroY = height - padding - ((0 - (-maxRes)) / (2 * maxRes)) * plotH;
    ctx.strokeStyle = '#94a3b8';
    ctx.lineWidth = 1.5;
    ctx.setLineDash([6, 4]);
    ctx.beginPath();
    ctx.moveTo(padding, zeroY);
    ctx.lineTo(width - padding, zeroY);
    ctx.stroke();
    ctx.setLineDash([]);

    // Scatter points
    for (let i = 0; i < residuals.length; i++) {
      const cx = padding + ((yPred[i] - minPred) / predRange) * plotW;
      const cy = height - padding - ((residuals[i] - (-maxRes)) / (2 * maxRes)) * plotH;
      const color = residuals[i] > 0 ? '#f59e0b' : '#ef4444';
      ctx.beginPath();
      ctx.arc(cx, cy, 3, 0, 2 * Math.PI);
      ctx.fillStyle = color;
      ctx.globalAlpha = 0.6;
      ctx.fill();
      ctx.globalAlpha = 1;
    }

    // Labels
    ctx.fillStyle = '#374151';
    ctx.font = '12px system-ui';
    ctx.textAlign = 'center';
    ctx.fillText('Valor Predicho', width / 2, height - 8);
    ctx.save();
    ctx.translate(14, height / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText('Residuo', 0, 0);
    ctx.restore();

    // Y-axis labels
    ctx.fillStyle = '#6b7280';
    ctx.font = '10px system-ui';
    ctx.textAlign = 'right';
    for (let i = 0; i <= 4; i++) {
      const val = -maxRes + (i / 4) * 2 * maxRes;
      const y = height - padding - (i / 4) * plotH;
      ctx.fillText(val.toFixed(1), padding - 8, y + 3);
    }

    ctx.fillStyle = '#94a3b8';
    ctx.font = '10px system-ui';
    ctx.textAlign = 'left';
    ctx.fillText('--- Residuo = 0', width - 120, zeroY - 8);
  }, [yTest, yPred]);

  return (
    <div className="flex flex-col items-center">
      <p className="text-sm font-semibold mb-2 text-foreground">Gráfico de Residuos</p>
      <canvas ref={canvasRef} className="rounded-lg border bg-white" style={{ width: 400, height: 300 }} />
    </div>
  );
}

// ============================================================
// Component: MetricCard (updated for regression support)
// ============================================================
function MetricCard({
  label,
  value,
  icon,
  color,
  isPercentage = true,
}: {
  label: string;
  value: number | undefined;
  icon: React.ReactNode;
  color: string;
  isPercentage?: boolean;
}) {
  const displayValue = value !== undefined
    ? isPercentage
      ? (value * 100).toFixed(1) + '%'
      : value.toFixed(4)
    : '—';
  const isGood = value !== undefined && (isPercentage ? value >= 0.8 : value >= 0);
  const isBad = value !== undefined && (isPercentage ? value < 0.6 : false);

  return (
    <Card className="relative overflow-hidden">
      <div className={`absolute top-0 left-0 w-1 h-full ${color}`} />
      <CardContent className="p-3 pl-4">
        <div className="flex items-center justify-between">
          <div>
            <p className="text-xs text-muted-foreground">{label}</p>
            <p className="text-xl font-bold font-mono">{displayValue}</p>
          </div>
          <div className="flex flex-col items-end gap-1">
            {icon}
            {value !== undefined &&
              (isGood ? (
                <CheckCircle2 className="h-4 w-4 text-green-500" />
              ) : isBad ? (
                <XCircle className="h-4 w-4 text-red-500" />
              ) : (
                <AlertTriangle className="h-4 w-4 text-yellow-500" />
              ))}
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

// ============================================================
// Component: F1BarChart (SVG horizontal bar for comparison)
// ============================================================
function F1BarChart({ results }: { results: ComparisonResult[] }) {
  if (results.length === 0) return null;
  const maxF1 = Math.max(...results.map((r) => r.metrics.f1), 0.01);
  const barHeight = 32;
  const chartWidth = 300;
  const labelWidth = 120;
  const totalWidth = labelWidth + chartWidth + 80;
  const totalHeight = results.length * (barHeight + 12) + 20;

  return (
    <div className="flex flex-col items-center">
      <p className="text-sm font-semibold mb-2 text-foreground">Comparación de F1-Score</p>
      <svg width={totalWidth} height={totalHeight}>
        {results.map((item, i) => {
          const barW = maxF1 > 0 ? (item.metrics.f1 / maxF1) * chartWidth : 0;
          const y = i * (barHeight + 12) + 5;
          const isBest = item.metrics.f1 === Math.max(...results.map((r) => r.metrics.f1));
          return (
            <g key={item.algorithmId}>
              <text
                x={labelWidth - 5}
                y={y + barHeight / 2 + 4}
                textAnchor="end"
                className="text-xs fill-foreground font-medium"
                fontSize={11}
              >
                {item.algorithmName.length > 16 ? item.algorithmName.substring(0, 14) + '…' : item.algorithmName}
              </text>
              <rect
                x={labelWidth}
                y={y}
                width={barW}
                height={barHeight}
                fill={item.color}
                rx={6}
                opacity={isBest ? 1 : 0.6}
              />
              <text
                x={labelWidth + barW + 8}
                y={y + barHeight / 2 + 5}
                className="text-xs fill-foreground font-bold"
                fontSize={12}
              >
                {(item.metrics.f1 * 100).toFixed(1)}%
                {isBest && ' ★'}
              </text>
            </g>
          );
        })}
      </svg>
    </div>
  );
}

// ============================================================
// Main Page Component
// ============================================================
export default function MLHyperLab() {
  // ── Lab state ──
  const [selectedAlgo, setSelectedAlgo] = useState<string>('xgboost');
  const [selectedDataset, setSelectedDataset] = useState<string>('moons');
  const [hyperparams, setHyperparams] = useState<Record<string, number | string | null>>(() =>
    getInitialHyperparams(ALGORITHMS[0])
  );
  const [isTraining, setIsTraining] = useState(false);
  const [trainResult, setTrainResult] = useState<TrainResult | null>(null);
  const [boundaryData, setBoundaryData] = useState<BoundaryData | null>(null);
  const [showExplanation, setShowExplanation] = useState(false);
  const [vizTab, setVizTab] = useState('boundary');

  // ── Comparison state ──
  const [compDataset, setCompDataset] = useState<string>('moons');
  const [compSelectedAlgos, setCompSelectedAlgos] = useState<Set<string>>(
    new Set(['xgboost', 'random_forest', 'knn'])
  );
  const [compResults, setCompResults] = useState<ComparisonResult[]>([]);
  const [compTraining, setCompTraining] = useState(false);
  const [compTrainingAlgo, setCompTrainingAlgo] = useState<string | null>(null);
  const [compTaskType, setCompTaskType] = useState<'classification' | 'regression'>('classification');
  const [compUseCustomDataset, setCompUseCustomDataset] = useState(false);

  // ── Workshops state ──
  const [selectedWorkshop, setSelectedWorkshop] = useState<Workshop | null>(null);
  const [currentStepIdx, setCurrentStepIdx] = useState(0);
  const [completedSteps, setCompletedSteps] = useState<Set<number>>(new Set());
  const [showHint, setShowHint] = useState(false);
  const [wsTrainResult, setWsTrainResult] = useState<TrainResult | null>(null);
  const [wsBoundaryData, setWsBoundaryData] = useState<BoundaryData | null>(null);
  const [wsTraining, setWsTraining] = useState(false);

  // ── Main tab ──
  const [mainTab, setMainTab] = useState('lab');

  // ── CSV Import state ──
  const [showCSVModal, setShowCSVModal] = useState(false);
  const [csvImportStep, setCsvImportStep] = useState<'upload' | 'configure' | 'preview'>('upload');
  const [csvParseProgress, setCsvParseProgress] = useState<ParseProgress | null>(null);
  const [csvColumns, setCsvColumns] = useState<string[]>([]);
  const [csvRows, setCsvRows] = useState<string[][]>([]);
  const [csvColumnTypes, setCsvColumnTypes] = useState<ColumnTypeInfo[]>([]);
  const [csvPreview, setCsvPreview] = useState<string[][]>([]);
  const [csvTargetColumn, setCsvTargetColumn] = useState<string>('');
  const [csvFileName, setCsvFileName] = useState<string>('');
  const [customDataset, setCustomDataset] = useState<ImportedDataset | null>(null);
  const [useCustomDataset, setUseCustomDataset] = useState(false);
  const [csvDragOver, setCsvDragOver] = useState(false);
  const [csvTaskTypeOverride, setCsvTaskTypeOverride] = useState<'auto' | 'classification' | 'regression'>('auto');

  // ── Task type (classification / regression) ──
  const [taskType, setTaskType] = useState<'classification' | 'regression'>('classification');
  const [autoDetectedType, setAutoDetectedType] = useState<'classification' | 'regression' | null>(null);

  // ── Regression state ──
  const [regTrainResult, setRegTrainResult] = useState<RegressionTrainResultLocal | null>(null);
  const [regPlotData, setRegPlotData] = useState<RegressionPlotDataLocal | null>(null);
  const [selectedRegAlgo, setSelectedRegAlgo] = useState<string>('xgboost_regressor');
  const [selectedRegDataset, setSelectedRegDataset] = useState<string>('make_regression');
  const [regHyperparams, setRegHyperparams] = useState<Record<string, number | string | null>>(() =>
    getInitialHyperparams(REGRESSION_ALGORITHMS[0])
  );
  const [regVizTab, setRegVizTab] = useState('scatter');
  const [regCompDataset, setRegCompDataset] = useState<string>('make_regression');
  const [regCompSelectedAlgos, setRegCompSelectedAlgos] = useState<Set<string>>(
    new Set(['xgboost_regressor', 'random_forest_regressor', 'linear_regression'])
  );
  const [regCompResults, setRegCompResults] = useState<RegressionComparisonResult[]>([]);
  const [regCompTraining, setRegCompTraining] = useState(false);

  const currentAlgo = taskType === 'regression'
    ? REGRESSION_ALGORITHMS.find((a) => a.id === selectedRegAlgo) ?? REGRESSION_ALGORITHMS[0]
    : ALGORITHMS.find((a) => a.id === selectedAlgo) ?? ALGORITHMS[0];
  const currentDataset = taskType === 'regression'
    ? REGRESSION_DATASET_PRESETS.find((d) => d.id === selectedRegDataset) ?? REGRESSION_DATASET_PRESETS[0]
    : DATASET_PRESETS.find((d) => d.id === selectedDataset) ?? DATASET_PRESETS[0];

  const handleAlgoChange = useCallback((algoId: string) => {
    setSelectedAlgo(algoId);
    const algo = ALGORITHMS.find((a) => a.id === algoId)!;
    setHyperparams(getInitialHyperparams(algo));
    setTrainResult(null);
    setBoundaryData(null);
  }, []);

  const handleDatasetChange = useCallback((datasetId: string) => {
    setSelectedDataset(datasetId);
    setTrainResult(null);
    setBoundaryData(null);
  }, []);

  const handleHyperparamChange = useCallback((key: string, value: number | string | null) => {
    setHyperparams((prev) => ({ ...prev, [key]: value }));
  }, []);

  const handleResetParams = useCallback(() => {
    setHyperparams(getInitialHyperparams(currentAlgo));
  }, [currentAlgo]);

  // ── CSV Import handlers ──
  const handleCSVFile = useCallback(async (file: File) => {
    setCsvFileName(file.name);
    setCsvImportStep('upload');
    setCsvParseProgress({ stage: 'reading', progress: 0, message: 'Leyendo archivo...' });

    try {
      const result = await parseCSVFile(file, (progress) => setCsvParseProgress(progress));
      setCsvColumns(result.columns);
      setCsvRows(result.rows);
      setCsvColumnTypes(result.columnTypes);
      setCsvPreview(result.preview);
      setCsvTargetColumn(suggestTargetColumn(result.columns, result.columnTypes));
      setCsvTaskTypeOverride('auto');
      setCsvImportStep('configure');
    } catch {
      setCsvParseProgress({ stage: 'error', progress: 0, message: 'Error al leer el archivo' });
    }
  }, []);

  const handleCSVDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setCsvDragOver(false);
    const file = e.dataTransfer.files[0];
    if (file && (file.name.endsWith('.csv') || file.name.endsWith('.txt') || file.name.endsWith('.tsv'))) {
      handleCSVFile(file);
    }
  }, [handleCSVFile]);

  const handleCSVConfirm = useCallback(() => {
    if (!csvTargetColumn || csvColumns.length === 0) return;

    // Determine task type: user override or auto-detect
    const autoDetected = detectTaskType(csvTargetColumn, csvColumnTypes, csvColumns, csvRows);
    const effectiveTaskType = csvTaskTypeOverride === 'auto' ? autoDetected : csvTaskTypeOverride;

    const dataset = prepareDataset(
      csvColumns, csvRows, csvColumnTypes, csvTargetColumn,
      csvFileName.replace(/\.(csv|txt|tsv)$/i, ''),
      csvFileName,
      10000,
      csvTaskTypeOverride === 'auto' ? undefined : csvTaskTypeOverride
    );

    setCustomDataset(dataset);
    setUseCustomDataset(true);
    setAutoDetectedType(autoDetected);

    // Set task type based on effective type (user override or auto)
    if (effectiveTaskType === 'regression') {
      setTaskType('regression');
      setSelectedRegDataset('__custom__');
      setCompTaskType('regression');
      setCompUseCustomDataset(true);
    } else {
      setTaskType('classification');
      setSelectedDataset('__custom__');
      setCompTaskType('classification');
      setCompUseCustomDataset(true);
    }

    setTrainResult(null);
    setBoundaryData(null);
    setRegTrainResult(null);
    setRegPlotData(null);
    setCompResults([]);
    setRegCompResults([]);
    setShowCSVModal(false);
    setCsvImportStep('upload');
  }, [csvColumns, csvRows, csvColumnTypes, csvTargetColumn, csvFileName, csvTaskTypeOverride]);

  const handleRemoveCustomDataset = useCallback(() => {
    setCustomDataset(null);
    setUseCustomDataset(false);
    setCompUseCustomDataset(false);
    setAutoDetectedType(null);
    setSelectedDataset('moons');
    setSelectedRegDataset('make_regression');
    setTrainResult(null);
    setBoundaryData(null);
    setRegTrainResult(null);
    setRegPlotData(null);
    setCompResults([]);
    setRegCompResults([]);
  }, []);

  // ── Regression handlers ──
  const handleRegAlgoChange = useCallback((algoId: string) => {
    setSelectedRegAlgo(algoId);
    const algo = REGRESSION_ALGORITHMS.find((a) => a.id === algoId)!;
    setRegHyperparams(getInitialHyperparams(algo));
    setRegTrainResult(null);
    setRegPlotData(null);
  }, []);

  const handleRegDatasetChange = useCallback((datasetId: string) => {
    setSelectedRegDataset(datasetId);
    setRegTrainResult(null);
    setRegPlotData(null);
  }, []);

  const handleRegHyperparamChange = useCallback((key: string, value: number | string | null) => {
    setRegHyperparams((prev) => ({ ...prev, [key]: value }));
  }, []);

  const handleRegResetParams = useCallback(() => {
    const algo = REGRESSION_ALGORITHMS.find((a) => a.id === selectedRegAlgo)!;
    setRegHyperparams(getInitialHyperparams(algo));
  }, [selectedRegAlgo]);

  const handleTaskTypeChange = useCallback((newType: 'classification' | 'regression') => {
    setTaskType(newType);
    setTrainResult(null);
    setBoundaryData(null);
    setRegTrainResult(null);
    setRegPlotData(null);

    // If using custom dataset, re-prepare with the new task type
    if (useCustomDataset && customDataset && csvColumns.length > 0) {
      const reprepared = prepareDataset(
        csvColumns, csvRows, csvColumnTypes, csvTargetColumn,
        csvFileName.replace(/\.(csv|txt|tsv)$/i, ''),
        csvFileName,
        10000,
        newType
      );
      setCustomDataset(reprepared);
      setAutoDetectedType(detectTaskType(csvTargetColumn, csvColumnTypes, csvColumns, csvRows));
      if (newType === 'regression') {
        setSelectedRegDataset('__custom__');
      } else {
        setSelectedDataset('__custom__');
      }
    }
  }, [useCustomDataset, customDataset, csvColumns, csvRows, csvColumnTypes, csvTargetColumn, csvFileName]);

  // Train model in browser
  const handleTrain = useCallback(async () => {
    setIsTraining(true);

    if (taskType === 'regression') {
      setRegTrainResult(null);
      setRegPlotData(null);
      try {
        const cleanParams = prepareCleanParams(regHyperparams, selectedRegAlgo);
        let regResult: RegressionTrainResultLocal;

        if (useCustomDataset && customDataset) {
          await new Promise((resolve) => setTimeout(resolve, 50));
          const result = trainRegressionModelWithData(
            selectedRegAlgo, customDataset.X, customDataset.y,
            customDataset.feature_names, customDataset.target_names, cleanParams, 0.3
          );
          regResult = {
            success: result.success,
            metrics: result.metrics,
            feature_importance: result.feature_importance,
            tree_text: result.tree_text,
            n_train: result.n_train,
            n_test: result.n_test,
            target_names: result.target_names,
            feature_names: result.feature_names,
            y_test: result.y_test,
            y_pred: result.y_pred,
            error: result.error,
          };
          setRegTrainResult(regResult);

          if (result.success) {
            const featureIndices = [0, Math.min(1, customDataset.feature_names.length - 1)];
            await new Promise((resolve) => setTimeout(resolve, 50));
            const plotData = computeRegressionPlotDataWithData(
              selectedRegAlgo, customDataset.X, customDataset.y,
              customDataset.feature_names, cleanParams, featureIndices
            );
            setRegPlotData({
              points_x: plotData.points_x,
              points_y: plotData.points_y,
              points_true: plotData.points_true,
              points_pred: plotData.points_pred,
              feature_names: plotData.feature_names,
              regression_line_x: plotData.regression_line_x,
              regression_line_y: plotData.regression_line_y,
            });
          }
        } else {
          await new Promise((resolve) => setTimeout(resolve, 50));
          const result = trainRegressionModel(selectedRegAlgo, selectedRegDataset, cleanParams, 0.3);
          regResult = {
            success: result.success,
            metrics: result.metrics,
            feature_importance: result.feature_importance,
            tree_text: result.tree_text,
            n_train: result.n_train,
            n_test: result.n_test,
            target_names: result.target_names,
            feature_names: result.feature_names,
            y_test: result.y_test,
            y_pred: result.y_pred,
            error: result.error,
          };
          setRegTrainResult(regResult);

          if (result.success) {
            const featureIndices = currentDataset.bestFor2D
              ? [0, 1]
              : [0, Math.min(1, (result.feature_names?.length ?? 2) - 1)];
            await new Promise((resolve) => setTimeout(resolve, 50));
            const plotData = computeRegressionPlotData(
              selectedRegAlgo, selectedRegDataset, cleanParams, featureIndices
            );
            setRegPlotData({
              points_x: plotData.points_x,
              points_y: plotData.points_y,
              points_true: plotData.points_true,
              points_pred: plotData.points_pred,
              feature_names: plotData.feature_names,
              regression_line_x: plotData.regression_line_x,
              regression_line_y: plotData.regression_line_y,
            });
          }
        }
      } catch {
        setRegTrainResult({
          success: false,
          metrics: { mse: 0, rmse: 0, mae: 0, r2: 0 },
          feature_importance: [],
          tree_text: null,
          n_train: 0,
          n_test: 0,
          target_names: [],
          feature_names: [],
          y_test: [],
          y_pred: [],
          error: 'Error al entrenar. Intenta de nuevo.',
        });
      } finally {
        setIsTraining(false);
      }
      return;
    }

    // Classification training (existing logic)
    setTrainResult(null);
    setBoundaryData(null);

    try {
      const cleanParams = prepareCleanParams(hyperparams, selectedAlgo);
      let trainResultData: TrainResult;

      if (useCustomDataset && customDataset) {
        await new Promise((resolve) => setTimeout(resolve, 50));
        trainResultData = trainModelWithData(
          selectedAlgo, customDataset.X, customDataset.y,
          customDataset.feature_names, customDataset.target_names,
          cleanParams, 0.3
        );
        setTrainResult(trainResultData);

        if (trainResultData.success) {
          const featureIndices = [0, Math.min(1, customDataset.feature_names.length - 1)];
          await new Promise((resolve) => setTimeout(resolve, 50));
          const boundary = computeDecisionBoundaryWithData(
            selectedAlgo, customDataset.X, customDataset.y,
            customDataset.feature_names, customDataset.target_names,
            cleanParams, featureIndices, 50
          );
          setBoundaryData(boundary);
        }
      } else {
        await new Promise((resolve) => setTimeout(resolve, 50));
        trainResultData = trainModel(selectedAlgo, selectedDataset, cleanParams, 0.3);
        setTrainResult(trainResultData);

        if (trainResultData.success) {
          const featureIndices = currentDataset.bestFor2D
            ? [0, 1]
            : [0, Math.min(1, (trainResultData.feature_names?.length ?? 2) - 1)];
          await new Promise((resolve) => setTimeout(resolve, 50));
          const boundary = computeDecisionBoundary(
            selectedAlgo, selectedDataset, cleanParams, featureIndices, 50
          );
          setBoundaryData(boundary);
        }
      }
    } catch {
      setTrainResult({
        success: false,
        metrics: { accuracy: 0, precision: 0, recall: 0, f1: 0 },
        confusion_matrix: [],
        feature_importance: [],
        tree_text: null,
        n_train: 0,
        n_test: 0,
        target_names: [],
        feature_names: [],
        error: 'Error al entrenar. Intenta de nuevo.',
      });
    } finally {
      setIsTraining(false);
    }
  }, [selectedAlgo, selectedDataset, hyperparams, currentDataset, useCustomDataset, customDataset, taskType, selectedRegAlgo, selectedRegDataset, regHyperparams]);

  // ── Comparison train ──
  const handleCompareTrain = useCallback(async () => {
    if (compTaskType === 'regression') {
      // ── Regression comparison ──
      setRegCompTraining(true);
      setRegCompResults([]);
      const results: RegressionComparisonResult[] = [];
      const algoColors = ['#10b981', '#f59e0b', '#8b5cf6', '#ef4444', '#06b6d4', '#ec4899'];
      const algosToTrain = REGRESSION_ALGORITHMS.filter((a) => regCompSelectedAlgos.has(a.id));

      for (let idx = 0; idx < algosToTrain.length; idx++) {
        const algo = algosToTrain[idx];
        setCompTrainingAlgo(algo.name);

        try {
          await new Promise((resolve) => setTimeout(resolve, 30));
          const defaultParams: Record<string, unknown> = {};
          algo.hyperparams.forEach((hp) => {
            defaultParams[hp.key] = hp.defaultValue;
          });

          let trainRes: RegressionTrainResultLocal;
          if (compUseCustomDataset && customDataset) {
            const result = trainRegressionModelWithData(
              algo.id, customDataset.X, customDataset.y,
              customDataset.feature_names, customDataset.target_names, defaultParams, 0.3
            );
            trainRes = {
              success: result.success,
              metrics: result.metrics,
              feature_importance: result.feature_importance,
              tree_text: result.tree_text,
              n_train: result.n_train,
              n_test: result.n_test,
              target_names: result.target_names,
              feature_names: result.feature_names,
              y_test: result.y_test,
              y_pred: result.y_pred,
              error: result.error,
            };
          } else {
            const result = trainRegressionModel(algo.id, regCompDataset, defaultParams, 0.3);
            trainRes = {
              success: result.success,
              metrics: result.metrics,
              feature_importance: result.feature_importance,
              tree_text: result.tree_text,
              n_train: result.n_train,
              n_test: result.n_test,
              target_names: result.target_names,
              feature_names: result.feature_names,
              y_test: result.y_test,
              y_pred: result.y_pred,
              error: result.error,
            };
          }

          if (trainRes.success && trainRes.metrics) {
            results.push({
              algorithmId: algo.id,
              algorithmName: algo.name,
              color: algoColors[idx % algoColors.length],
              metrics: trainRes.metrics,
              featureImportance: trainRes.feature_importance || [],
              n_train: trainRes.n_train || 0,
              n_test: trainRes.n_test || 0,
            });
          }
        } catch {
          // skip failed models
        }
      }

      setRegCompResults(results);
      setRegCompTraining(false);
      setCompTrainingAlgo(null);
      return;
    }

    // ── Classification comparison ──
    setCompTraining(true);
    setCompResults([]);
    const results: ComparisonResult[] = [];
    const algoColors = ['#10b981', '#f59e0b', '#8b5cf6', '#ef4444', '#06b6d4', '#ec4899'];
    const algosToTrain = ALGORITHMS.filter((a) => compSelectedAlgos.has(a.id));

    for (let idx = 0; idx < algosToTrain.length; idx++) {
      const algo = algosToTrain[idx];
      setCompTrainingAlgo(algo.name);

      try {
        await new Promise((resolve) => setTimeout(resolve, 30));
        const defaultParams: Record<string, unknown> = {};
        algo.hyperparams.forEach((hp) => {
          if (
            hp.key === 'max_depth' &&
            (hp.defaultValue === 0 || hp.defaultValue === '0') &&
            (algo.id === 'decision_tree' || algo.id === 'random_forest')
          ) {
            defaultParams[hp.key] = null;
          } else {
            defaultParams[hp.key] = hp.defaultValue;
          }
        });

        let trainRes: TrainResult;
        let boundary: BoundaryData | null = null;

        if (compUseCustomDataset && customDataset) {
          // Use custom (CSV) dataset
          trainRes = trainModelWithData(
            algo.id, customDataset.X, customDataset.y,
            customDataset.feature_names, customDataset.target_names,
            defaultParams, 0.3
          );
          if (trainRes.success) {
            const featureIndices = [0, Math.min(1, customDataset.feature_names.length - 1)];
            await new Promise((resolve) => setTimeout(resolve, 30));
            boundary = computeDecisionBoundaryWithData(
              algo.id, customDataset.X, customDataset.y,
              customDataset.feature_names, customDataset.target_names,
              defaultParams, featureIndices, 40
            );
          }
        } else {
          // Use built-in dataset
          trainRes = trainModel(algo.id, compDataset, defaultParams, 0.3);
          if (trainRes.success) {
            const ds = DATASET_PRESETS.find((d) => d.id === compDataset)!;
            const featureIndices = ds.bestFor2D
              ? [0, 1]
              : [0, Math.min(1, (trainRes.feature_names?.length ?? 2) - 1)];
            await new Promise((resolve) => setTimeout(resolve, 30));
            boundary = computeDecisionBoundary(algo.id, compDataset, defaultParams, featureIndices, 40);
          }
        }

        if (trainRes.success && trainRes.metrics) {
          results.push({
            algorithmId: algo.id,
            algorithmName: algo.name,
            color: algoColors[idx % algoColors.length],
            metrics: trainRes.metrics,
            confusion_matrix: trainRes.confusion_matrix,
            boundaryData: boundary,
            featureImportance: trainRes.feature_importance || [],
            n_train: trainRes.n_train || 0,
            n_test: trainRes.n_test || 0,
          });
        }
      } catch {
        // skip failed models
      }
    }

    setCompResults(results);
    setCompTraining(false);
    setCompTrainingAlgo(null);
  }, [compDataset, compSelectedAlgos, compTaskType, compUseCustomDataset, customDataset, regCompDataset, regCompSelectedAlgos]);

  // ── Workshop helpers ──
  const handleSelectWorkshop = useCallback((ws: Workshop) => {
    setSelectedWorkshop(ws);
    setCurrentStepIdx(0);
    setCompletedSteps(new Set());
    setShowHint(false);
    setWsTrainResult(null);
    setWsBoundaryData(null);
  }, []);

  const handleConfigureStep = useCallback((step: WorkshopStep) => {
    setSelectedAlgo(step.algorithm);
    const algo = ALGORITHMS.find((a) => a.id === step.algorithm)!;
    const params: Record<string, number | string | null> = {};
    algo.hyperparams.forEach((hp) => {
      if (step.hyperparams[hp.key] !== undefined) {
        const val = step.hyperparams[hp.key];
        if (hp.key === 'max_depth' && (val === 0 || val === '0') && (step.algorithm === 'decision_tree' || step.algorithm === 'random_forest')) {
          params[hp.key] = null;
        } else {
          params[hp.key] = val;
        }
      } else {
        params[hp.key] = hp.defaultValue;
      }
    });
    setHyperparams(params);
    setSelectedDataset(step.dataset);
    setTrainResult(null);
    setBoundaryData(null);
    setWsTrainResult(null);
    setWsBoundaryData(null);
    setMainTab('lab');
  }, []);

  const handleWorkshopTrain = useCallback(async (step: WorkshopStep) => {
    setWsTraining(true);
    setWsTrainResult(null);
    setWsBoundaryData(null);

    try {
      const algo = ALGORITHMS.find((a) => a.id === step.algorithm)!;
      const cleanParams: Record<string, unknown> = {};
      algo.hyperparams.forEach((hp) => {
        if (step.hyperparams[hp.key] !== undefined) {
          const val = step.hyperparams[hp.key];
          if (hp.key === 'max_depth' && (val === 0 || val === '0') && (step.algorithm === 'decision_tree' || step.algorithm === 'random_forest')) {
            cleanParams[hp.key] = null;
          } else {
            cleanParams[hp.key] = val;
          }
        } else {
          cleanParams[hp.key] = hp.defaultValue;
        }
      });

      await new Promise((resolve) => setTimeout(resolve, 50));
      const trainRes = trainModel(step.algorithm, step.dataset, cleanParams, 0.3);
      setWsTrainResult(trainRes);

      if (trainRes.success) {
        const ds = DATASET_PRESETS.find((d) => d.id === step.dataset)!;
        const featureIndices = ds.bestFor2D
          ? [0, 1]
          : [0, Math.min(1, (trainRes.feature_names?.length ?? 2) - 1)];
        await new Promise((resolve) => setTimeout(resolve, 50));
        const boundary = computeDecisionBoundary(step.algorithm, step.dataset, cleanParams, featureIndices, 50);
        setWsBoundaryData(boundary);
      }
    } catch {
      setWsTrainResult({
        success: false,
        metrics: { accuracy: 0, precision: 0, recall: 0, f1: 0 },
        confusion_matrix: [],
        feature_importance: [],
        tree_text: null,
        n_train: 0,
        n_test: 0,
        target_names: [],
        feature_names: [],
        error: 'Error al entrenar.',
      });
    } finally {
      setWsTraining(false);
    }
  }, []);

  const handleCompleteStep = useCallback(() => {
    setCompletedSteps((prev) => new Set([...prev, currentStepIdx]));
  }, [currentStepIdx]);

  // ── Difficulty badge colors ──
  const diffColors: Record<string, string> = {
    principiante: 'bg-emerald-100 text-emerald-700 border-emerald-200',
    intermedio: 'bg-amber-100 text-amber-700 border-amber-200',
    avanzado: 'bg-red-100 text-red-700 border-red-200',
  };

  // ── Best metric helpers for comparison ──
  const getBestMetricKey = (metricKey: keyof Metrics) => {
    if (compResults.length === 0) return null;
    const best = compResults.reduce((a, b) =>
      a.metrics[metricKey] >= b.metrics[metricKey] ? a : b
    );
    return best.algorithmId;
  };

  // ────────────────────────────────────────────────────────────
  // RENDER
  // ────────────────────────────────────────────────────────────
  return (
    <TooltipProvider>
      <div className="min-h-screen bg-gradient-to-br from-slate-50 via-white to-emerald-50/30 flex flex-col">
        {/* Header */}
        <header className="sticky top-0 z-50 bg-white/80 backdrop-blur-md border-b">
          <div className="max-w-7xl mx-auto px-4 py-3">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <div className="h-9 w-9 rounded-lg bg-gradient-to-br from-emerald-500 to-teal-600 flex items-center justify-center">
                  <Brain className="h-5 w-5 text-white" />
                </div>
                <div>
                  <h1 className="text-lg font-bold text-foreground tracking-tight">
                    ML HyperLab
                  </h1>
                  <p className="text-xs text-muted-foreground">
                    Laboratorio Interactivo de Machine Learning
                  </p>
                </div>
              </div>
              {mainTab === 'lab' && (
                <div className="flex items-center gap-2">
                  <Badge variant="outline" className={`text-xs ${taskType === 'regression' ? 'border-blue-300 text-blue-600' : ''}`}>
                    {currentAlgo.emoji} {currentAlgo.name}
                  </Badge>
                  <Badge variant="secondary" className={`text-xs gap-1 ${taskType === 'regression' ? 'bg-blue-100 text-blue-700' : ''}`}>
                    {taskType === 'regression' ? '📉 Regresión' : '🎯 Clasificación'}
                  </Badge>
                  {useCustomDataset && customDataset ? (
                    <Badge variant="secondary" className="text-xs gap-1">
                      📁 {customDataset.name}
                    </Badge>
                  ) : (
                    <Badge variant="secondary" className="text-xs">
                      {currentDataset.emoji} {currentDataset.name}
                    </Badge>
                  )}
                </div>
              )}
            </div>
          </div>
        </header>

        <main className="max-w-7xl mx-auto px-4 py-6 flex-1">
          {/* Main Tabs */}
          <Tabs value={mainTab} onValueChange={setMainTab} className="mb-6">
            <TabsList className="grid w-full grid-cols-3 max-w-md mx-auto">
              <TabsTrigger value="lab" className="text-sm gap-1.5">
                🧪 <span className="hidden sm:inline">Laboratorio</span>
              </TabsTrigger>
              <TabsTrigger value="comparison" className="text-sm gap-1.5">
                📊 <span className="hidden sm:inline">Comparación</span>
              </TabsTrigger>
              <TabsTrigger value="workshops" className="text-sm gap-1.5">
                📚 <span className="hidden sm:inline">Talleres</span>
              </TabsTrigger>
            </TabsList>

            {/* ═══════════════════════════════════════════════════════
                TAB: LABORATORIO
                ═══════════════════════════════════════════════════════ */}
            <TabsContent value="lab" className="mt-6">
              {/* Task Type Toggle */}
              <div className="flex items-center justify-center mb-4 gap-3">
                <div className="inline-flex items-center rounded-lg border bg-muted p-1 gap-1">
                  <button
                    onClick={() => handleTaskTypeChange('classification')}
                    className={`inline-flex items-center gap-1.5 rounded-md px-4 py-2 text-sm font-medium transition-all ${
                      taskType === 'classification'
                        ? 'bg-emerald-500 text-white shadow-sm'
                        : 'text-muted-foreground hover:text-foreground'
                    }`}
                  >
                    <Target className="h-4 w-4" />
                    Clasificación
                  </button>
                  <button
                    onClick={() => handleTaskTypeChange('regression')}
                    className={`inline-flex items-center gap-1.5 rounded-md px-4 py-2 text-sm font-medium transition-all ${
                      taskType === 'regression'
                        ? 'bg-blue-500 text-white shadow-sm'
                        : 'text-muted-foreground hover:text-foreground'
                    }`}
                  >
                    <TrendingUp className="h-4 w-4" />
                    Regresión
                  </button>
                </div>
                {useCustomDataset && autoDetectedType && autoDetectedType !== taskType && (
                  <TooltipProvider delayDuration={200}>
                    <Tooltip>
                      <TooltipTrigger asChild>
                        <Badge variant="outline" className="text-[10px] gap-1 border-amber-300 text-amber-700 bg-amber-50 cursor-help">
                          <AlertTriangle className="h-3 w-3" />
                          Override activo
                        </Badge>
                      </TooltipTrigger>
                      <TooltipContent className="max-w-xs p-3">
                        <p className="text-xs font-medium mb-1">Auto-detección sobreescrita</p>
                        <p className="text-xs text-muted-foreground">
                          El sistema detectó <strong>{autoDetectedType === 'classification' ? 'Clasificación' : 'Regresión'}</strong> automáticamente,
                          pero estás usando <strong>{taskType === 'classification' ? 'Clasificación' : 'Regresión'}</strong>.
                          Puedes cambiar esto en cualquier momento con el selector de arriba.
                        </p>
                      </TooltipContent>
                    </Tooltip>
                  </TooltipProvider>
                )}
              </div>

              {/* Algorithm Selector */}
              <section className="mb-6">
                <h2 className="text-sm font-semibold text-muted-foreground uppercase tracking-wider mb-3">
                  Selecciona un Algoritmo
                </h2>
                <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3">
                  {(taskType === 'classification' ? ALGORITHMS : REGRESSION_ALGORITHMS).map((algo) => (
                    <button
                      key={algo.id}
                      onClick={() => {
                        if (taskType === 'regression') handleRegAlgoChange(algo.id);
                        else handleAlgoChange(algo.id);
                      }}
                      className={`relative p-3 rounded-xl border-2 transition-all text-left ${
                        (taskType === 'regression' ? selectedRegAlgo : selectedAlgo) === algo.id
                          ? taskType === 'regression'
                            ? 'border-blue-500 bg-blue-50/50 shadow-md'
                            : 'border-emerald-500 bg-emerald-50/50 shadow-md'
                          : 'border-border hover:border-blue-300 hover:shadow-sm bg-white'
                      }`}
                    >
                      <div className="text-2xl mb-1">{algo.emoji}</div>
                      <p className="text-sm font-semibold text-foreground">{algo.shortName}</p>
                      <p className="text-[10px] text-muted-foreground leading-tight mt-0.5">
                        {algo.name}
                      </p>
                      {(taskType === 'regression' ? selectedRegAlgo : selectedAlgo) === algo.id && (
                        <div className="absolute top-2 right-2">
                          <CheckCircle2 className={`h-4 w-4 ${taskType === 'regression' ? 'text-blue-500' : 'text-emerald-500'}`} />
                        </div>
                      )}
                    </button>
                  ))}
                </div>
              </section>

              {/* Algorithm Explanation */}
              <Card className="mb-6 overflow-hidden">
                <div className="h-1.5" style={{ backgroundColor: currentAlgo.color }} />
                <CardHeader className="pb-2">
                  <div className="flex items-center justify-between">
                    <CardTitle className="text-lg">
                      {currentAlgo.emoji} {currentAlgo.name}
                    </CardTitle>
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => setShowExplanation(!showExplanation)}
                      className="text-xs"
                    >
                      <BookOpen className="h-4 w-4 mr-1" />
                      {showExplanation ? 'Ocultar' : 'Explicación completa'}
                    </Button>
                  </div>
                  <CardDescription>{currentAlgo.description}</CardDescription>
                </CardHeader>
                {showExplanation && (
                  <CardContent className="pt-0">
                    <Separator className="my-3" />
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      <div>
                        <h4 className="text-sm font-semibold mb-2 flex items-center gap-1">
                          <Lightbulb className="h-4 w-4 text-yellow-500" />
                          Concepto
                        </h4>
                        <p className="text-sm text-muted-foreground leading-relaxed">
                          {currentAlgo.concept}
                        </p>
                      </div>
                      <div>
                        <h4 className="text-sm font-semibold mb-2 flex items-center gap-1">
                          <Target className="h-4 w-4 text-emerald-500" />
                          Cuándo usarlo
                        </h4>
                        <p className="text-sm text-muted-foreground leading-relaxed">
                          {currentAlgo.whenToUse}
                        </p>
                      </div>
                    </div>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-4">
                      <div>
                        <h4 className="text-sm font-semibold mb-2 flex items-center gap-1 text-green-600">
                          <CheckCircle2 className="h-4 w-4" /> Ventajas
                        </h4>
                        <ul className="space-y-1">
                          {currentAlgo.pros.map((p, i) => (
                            <li key={i} className="text-sm text-muted-foreground flex items-start gap-1.5">
                              <ChevronRight className="h-3.5 w-3.5 text-green-500 shrink-0 mt-0.5" />
                              {p}
                            </li>
                          ))}
                        </ul>
                      </div>
                      <div>
                        <h4 className="text-sm font-semibold mb-2 flex items-center gap-1 text-red-600">
                          <XCircle className="h-4 w-4" /> Desventajas
                        </h4>
                        <ul className="space-y-1">
                          {currentAlgo.cons.map((c, i) => (
                            <li key={i} className="text-sm text-muted-foreground flex items-start gap-1.5">
                              <ChevronRight className="h-3.5 w-3.5 text-red-500 shrink-0 mt-0.5" />
                              {c}
                            </li>
                          ))}
                        </ul>
                      </div>
                    </div>
                  </CardContent>
                )}
              </Card>

              {/* Main Content: Controls + Results */}
              <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
                {/* Left Panel */}
                <div className="lg:col-span-5 lg:sticky lg:top-20 lg:self-start flex flex-col gap-4 lg:max-h-[calc(100vh-7rem)]">
                  <Card>
                    <CardHeader className="pb-2">
                      <div className="flex items-center justify-between">
                        <CardTitle className="text-sm">Dataset</CardTitle>
                        <Button
                          variant="outline"
                          size="sm"
                          onClick={() => setShowCSVModal(true)}
                          className="text-xs h-7 gap-1"
                        >
                          <Upload className="h-3 w-3" />
                          Importar CSV
                        </Button>
                      </div>
                    </CardHeader>
                    <CardContent>
                      <div className="grid grid-cols-2 sm:grid-cols-3 gap-2">
                        {customDataset && (
                          <div className="relative p-2 rounded-lg border-2 border-blue-500 bg-blue-50/50 text-left transition-all">
                            <button
                              onClick={handleRemoveCustomDataset}
                              className="absolute top-1 right-1 h-5 w-5 rounded-full bg-red-100 hover:bg-red-200 flex items-center justify-center transition-colors"
                            >
                              <X className="h-3 w-3 text-red-500" />
                            </button>
                            <span className="text-lg">📁</span>
                            <p className="text-xs font-semibold mt-1 text-blue-700 truncate">
                              {customDataset.name}
                            </p>
                            <p className="text-[9px] text-muted-foreground leading-tight">
                              {customDataset.nSamples} filas · {customDataset.nFeatures} features · {customDataset.taskType === 'regression' ? 'Regresión' : `${customDataset.nClasses} clases`}
                            </p>
                          </div>
                        )}
                        {(taskType === 'classification' ? DATASET_PRESETS : REGRESSION_DATASET_PRESETS).map((ds) => (
                          <button
                            key={ds.id}
                            onClick={() => {
                              if (taskType === 'regression') {
                                handleRegDatasetChange(ds.id);
                              } else {
                                handleDatasetChange(ds.id);
                              }
                              if (useCustomDataset) {
                                setUseCustomDataset(false);
                                setCustomDataset(null);
                              }
                            }}
                            className={`p-2 rounded-lg border text-left transition-all ${
                              (taskType === 'regression' ? selectedRegDataset : selectedDataset) === ds.id && !useCustomDataset
                                ? taskType === 'regression'
                                  ? 'border-blue-500 bg-blue-50/50'
                                  : 'border-emerald-500 bg-emerald-50/50'
                                : 'border-border hover:border-blue-300 bg-white'
                            }`}
                          >
                            <span className="text-lg">{ds.emoji}</span>
                            <p className="text-xs font-semibold mt-1">{ds.name}</p>
                            <p className="text-[9px] text-muted-foreground leading-tight">
                              {ds.description}
                            </p>
                          </button>
                        ))}
                      </div>
                    </CardContent>
                  </Card>

                  <Card className="flex-1 min-h-0 flex flex-col overflow-hidden">
                    <CardHeader className="pb-2 shrink-0">
                      <div className="flex items-center justify-between">
                        <CardTitle className="text-sm">Hiperparámetros</CardTitle>
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={taskType === 'regression' ? handleRegResetParams : handleResetParams}
                          className="text-xs h-7"
                        >
                          <RotateCcw className="h-3 w-3 mr-1" />
                          Resetear
                        </Button>
                      </div>
                      <CardDescription className="text-xs">
                        Ajusta los valores y observa cómo cambian los resultados. Haz clic en{' '}
                        <Info className="h-3 w-3 inline" /> para ver la explicación de cada parámetro.
                      </CardDescription>
                    </CardHeader>
                    <CardContent className="flex-1 min-h-0 overflow-hidden">
                      <ScrollArea className="h-full max-h-[400px] pr-2">
                        <div className="space-y-2">
                          {currentAlgo.hyperparams.map((hp) => (
                            <HyperparameterControlWrapper
                              key={hp.key}
                              config={hp}
                              value={taskType === 'regression' ? regHyperparams[hp.key] : hyperparams[hp.key]}
                              onChange={taskType === 'regression' ? handleRegHyperparamChange : handleHyperparamChange}
                              disabled={isTraining}
                              algoId={taskType === 'regression' ? selectedRegAlgo : selectedAlgo}
                            />
                          ))}
                        </div>
                      </ScrollArea>
                    </CardContent>
                  </Card>

                  <Button
                    onClick={handleTrain}
                    disabled={isTraining}
                    className={`w-full h-12 text-base font-semibold text-white shadow-lg shrink-0 ${
                      taskType === 'regression'
                        ? 'bg-gradient-to-r from-blue-500 to-indigo-600 hover:from-blue-600 hover:to-indigo-700'
                        : 'bg-gradient-to-r from-emerald-500 to-teal-600 hover:from-emerald-600 hover:to-teal-700'
                    }`}
                    size="lg"
                  >
                    {isTraining ? (
                      <>
                        <Loader2 className="h-5 w-5 mr-2 animate-spin" />
                        Entrenando modelo...
                      </>
                    ) : (
                      <>
                        <Play className="h-5 w-5 mr-2" />
                        Entrenar Modelo
                      </>
                    )}
                  </Button>
                </div>

                {/* Right Panel */}
                <div className="lg:col-span-7 space-y-4">
                  {/* Classification Results */}
                  {taskType === 'classification' && trainResult?.success && trainResult.metrics && (
                    <>
                      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                        <MetricCard
                          label="Accuracy"
                          value={trainResult.metrics.accuracy}
                          icon={<Target className="h-5 w-5 text-emerald-500" />}
                          color="bg-emerald-500"
                        />
                        <MetricCard
                          label="Precision"
                          value={trainResult.metrics.precision}
                          icon={<BarChart3 className="h-5 w-5 text-teal-500" />}
                          color="bg-teal-500"
                        />
                        <MetricCard
                          label="Recall"
                          value={trainResult.metrics.recall}
                          icon={<TrendingUp className="h-5 w-5 text-amber-500" />}
                          color="bg-amber-500"
                        />
                        <MetricCard
                          label="F1-Score"
                          value={trainResult.metrics.f1}
                          icon={<Zap className="h-5 w-5 text-violet-500" />}
                          color="bg-violet-500"
                        />
                      </div>
                      <div className="flex items-center gap-2 text-xs text-muted-foreground">
                        <span>
                          Train: {trainResult.n_train} | Test: {trainResult.n_test} muestras
                        </span>
                        <span>•</span>
                        <span>
                          {trainResult.target_names?.length} clases: {trainResult.target_names?.join(', ')}
                        </span>
                      </div>
                    </>
                  )}

                  {/* Regression Results */}
                  {taskType === 'regression' && regTrainResult?.success && regTrainResult.metrics && (
                    <>
                      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                        <MetricCard
                          label="MSE"
                          value={regTrainResult.metrics.mse}
                          icon={<Target className="h-5 w-5 text-blue-500" />}
                          color="bg-blue-500"
                          isPercentage={false}
                        />
                        <MetricCard
                          label="RMSE"
                          value={regTrainResult.metrics.rmse}
                          icon={<BarChart3 className="h-5 w-5 text-indigo-500" />}
                          color="bg-indigo-500"
                          isPercentage={false}
                        />
                        <MetricCard
                          label="MAE"
                          value={regTrainResult.metrics.mae}
                          icon={<TrendingUp className="h-5 w-5 text-amber-500" />}
                          color="bg-amber-500"
                          isPercentage={false}
                        />
                        <MetricCard
                          label="R²"
                          value={regTrainResult.metrics.r2}
                          icon={<Zap className="h-5 w-5 text-violet-500" />}
                          color="bg-violet-500"
                          isPercentage={false}
                        />
                      </div>
                      <div className="flex items-center gap-2 text-xs text-muted-foreground">
                        <span>
                          Train: {regTrainResult.n_train} | Test: {regTrainResult.n_test} muestras
                        </span>
                        <span>•</span>
                        <span>
                          Target: {regTrainResult.target_names?.join(', ') || 'valor'}
                        </span>
                      </div>
                    </>
                  )}

                  {(taskType === 'classification' ? trainResult : (taskType === 'regression' ? regTrainResult : null)) && !(taskType === 'classification' ? trainResult?.success : regTrainResult?.success) && (
                    <Card className="border-red-200 bg-red-50">
                      <CardContent className="p-4">
                        <div className="flex items-center gap-2 text-red-700">
                          <XCircle className="h-5 w-5" />
                          <p className="text-sm font-medium">Error al entrenar</p>
                        </div>
                        <p className="text-sm text-red-600 mt-1">
                          {(taskType === 'classification' ? trainResult?.error : regTrainResult?.error) || 'Error desconocido'}
                        </p>
                      </CardContent>
                    </Card>
                  )}

                  <Card>
                    <CardHeader className="pb-2">
                      <CardTitle className="text-sm">Visualizaciones</CardTitle>
                    </CardHeader>
                    <CardContent>
                      {taskType === 'classification' ? (
                        <Tabs value={vizTab} onValueChange={setVizTab}>
                          <TabsList className="w-full">
                            <TabsTrigger value="boundary" className="flex-1 text-xs">
                              Frontera de Decisión
                            </TabsTrigger>
                            <TabsTrigger value="confusion" className="flex-1 text-xs">
                              Matriz de Confusión
                            </TabsTrigger>
                            <TabsTrigger value="importance" className="flex-1 text-xs">
                              Importancia
                            </TabsTrigger>
                            {trainResult?.tree_text && (
                              <TabsTrigger value="tree" className="flex-1 text-xs">
                                Árbol
                              </TabsTrigger>
                            )}
                          </TabsList>
                          <TabsContent value="boundary" className="mt-4">
                            {isTraining ? (
                              <div className="flex items-center justify-center h-[400px]">
                                <Loader2 className="h-8 w-8 animate-spin text-emerald-500" />
                              </div>
                            ) : (
                              <DecisionBoundaryCanvas data={boundaryData} />
                            )}
                          </TabsContent>
                          <TabsContent value="confusion" className="mt-4">
                            {trainResult?.success && trainResult.confusion_matrix ? (
                              <ConfusionMatrixHeatmap
                                matrix={trainResult.confusion_matrix}
                                targetNames={trainResult.target_names || []}
                              />
                            ) : (
                              <div className="flex items-center justify-center h-[300px] bg-muted/30 rounded-lg border border-dashed">
                                <p className="text-sm text-muted-foreground">
                                  Entrena el modelo para ver la matriz de confusión
                                </p>
                              </div>
                            )}
                          </TabsContent>
                          <TabsContent value="importance" className="mt-4">
                            {trainResult?.success ? (
                              <FeatureImportanceChart
                                data={trainResult.feature_importance || []}
                                algorithmId={selectedAlgo}
                              />
                            ) : (
                              <div className="flex items-center justify-center h-[200px] bg-muted/30 rounded-lg border border-dashed">
                                <p className="text-sm text-muted-foreground">
                                  Entrena el modelo para ver la importancia
                                </p>
                              </div>
                            )}
                          </TabsContent>
                          {trainResult?.tree_text && (
                            <TabsContent value="tree" className="mt-4">
                              <ScrollArea className="max-h-[500px]">
                                <pre className="text-xs font-mono bg-muted/30 p-4 rounded-lg whitespace-pre-wrap">
                                  {trainResult.tree_text}
                                </pre>
                              </ScrollArea>
                            </TabsContent>
                          )}
                        </Tabs>
                      ) : (
                        /* Regression Visualizations */
                        <Tabs value={regVizTab} onValueChange={setRegVizTab}>
                          <TabsList className="w-full">
                            <TabsTrigger value="scatter" className="flex-1 text-xs">
                              Real vs Predicho
                            </TabsTrigger>
                            <TabsTrigger value="residuals" className="flex-1 text-xs">
                              Residuos
                            </TabsTrigger>
                            <TabsTrigger value="importance" className="flex-1 text-xs">
                              Importancia
                            </TabsTrigger>
                            {regTrainResult?.tree_text && (
                              <TabsTrigger value="tree" className="flex-1 text-xs">
                                Árbol
                              </TabsTrigger>
                            )}
                          </TabsList>
                          <TabsContent value="scatter" className="mt-4">
                            {isTraining ? (
                              <div className="flex items-center justify-center h-[400px]">
                                <Loader2 className="h-8 w-8 animate-spin text-blue-500" />
                              </div>
                            ) : (
                              <RegressionScatterCanvas data={regPlotData} />
                            )}
                          </TabsContent>
                          <TabsContent value="residuals" className="mt-4">
                            {regTrainResult?.success && regTrainResult.y_test && regTrainResult.y_pred ? (
                              <ResidualPlotCanvas yTest={regTrainResult.y_test} yPred={regTrainResult.y_pred} />
                            ) : (
                              <div className="flex items-center justify-center h-[300px] bg-muted/30 rounded-lg border border-dashed">
                                <p className="text-sm text-muted-foreground">
                                  Entrena el modelo para ver los residuos
                                </p>
                              </div>
                            )}
                          </TabsContent>
                          <TabsContent value="importance" className="mt-4">
                            {regTrainResult?.success ? (
                              <FeatureImportanceChart
                                data={regTrainResult.feature_importance || []}
                                algorithmId={selectedRegAlgo}
                              />
                            ) : (
                              <div className="flex items-center justify-center h-[200px] bg-muted/30 rounded-lg border border-dashed">
                                <p className="text-sm text-muted-foreground">
                                  Entrena el modelo para ver la importancia
                                </p>
                              </div>
                            )}
                          </TabsContent>
                          {regTrainResult?.tree_text && (
                            <TabsContent value="tree" className="mt-4">
                              <ScrollArea className="max-h-[500px]">
                                <pre className="text-xs font-mono bg-muted/30 p-4 rounded-lg whitespace-pre-wrap">
                                  {regTrainResult.tree_text}
                                </pre>
                              </ScrollArea>
                            </TabsContent>
                          )}
                        </Tabs>
                      )}
                    </CardContent>
                  </Card>

                  {!trainResult && !regTrainResult && !isTraining && (
                    <Card className="border-dashed">
                      <CardContent className="p-8">
                        <div className="text-center">
                          <Brain className={`h-16 w-16 mx-auto mb-4 ${taskType === 'regression' ? 'text-blue-300' : 'text-emerald-300'}`} />
                          <h3 className="text-lg font-semibold text-foreground mb-2">
                            Listo para experimentar
                          </h3>
                          <p className="text-sm text-muted-foreground max-w-md mx-auto">
                            {taskType === 'regression'
                              ? 'Selecciona un algoritmo de regresión, elige un dataset, ajusta los hiperparámetros y presiona "Entrenar Modelo" para ver los resultados y métricas como MSE, RMSE, MAE y R².'
                              : 'Selecciona un algoritmo, elige un dataset, ajusta los hiperparámetros y presiona "Entrenar Modelo" para ver los resultados en tiempo real.'}
                          </p>
                          <div className="mt-4 flex items-center justify-center gap-2 text-xs text-muted-foreground">
                            <Info className="h-3 w-3" /> Haz clic en el ícono de info junto a cada
                            parámetro para entender qué hace
                          </div>
                        </div>
                      </CardContent>
                    </Card>
                  )}
                </div>
              </div>
            </TabsContent>

            {/* ═══════════════════════════════════════════════════════
                TAB: COMPARACIÓN
                ═══════════════════════════════════════════════════════ */}
            <TabsContent value="comparison" className="mt-6">
              {/* Task Type Toggle for Comparison */}
              <div className="flex items-center justify-center mb-4">
                <div className="inline-flex items-center rounded-lg border bg-muted p-1 gap-1">
                  <button
                    onClick={() => { setCompTaskType('classification'); setCompResults([]); setRegCompResults([]); }}
                    className={`inline-flex items-center gap-1.5 rounded-md px-4 py-2 text-sm font-medium transition-all ${
                      compTaskType === 'classification'
                        ? 'bg-emerald-500 text-white shadow-sm'
                        : 'text-muted-foreground hover:text-foreground'
                    }`}
                  >
                    <Target className="h-4 w-4" />
                    Clasificación
                  </button>
                  <button
                    onClick={() => { setCompTaskType('regression'); setCompResults([]); setRegCompResults([]); }}
                    className={`inline-flex items-center gap-1.5 rounded-md px-4 py-2 text-sm font-medium transition-all ${
                      compTaskType === 'regression'
                        ? 'bg-blue-500 text-white shadow-sm'
                        : 'text-muted-foreground hover:text-foreground'
                    }`}
                  >
                    <TrendingUp className="h-4 w-4" />
                    Regresión
                  </button>
                </div>
              </div>

              {/* Config */}
              <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 mb-6">
                <div className="lg:col-span-4 space-y-4">
                  {/* Dataset selector */}
                  <Card>
                    <CardHeader className="pb-2">
                      <CardTitle className="text-sm flex items-center gap-2">
                        <BarChart3 className="h-4 w-4 text-emerald-500" />
                        Dataset para comparar
                      </CardTitle>
                    </CardHeader>
                    <CardContent>
                      {/* Custom dataset option */}
                      {customDataset && (
                        <div className="relative mb-2">
                          <button
                            onClick={() => {
                              // Auto-switch compTaskType to match dataset's task type
                              if (customDataset.taskType !== compTaskType) {
                                setCompTaskType(customDataset.taskType);
                              }
                              setCompUseCustomDataset(true);
                              setCompResults([]);
                              setRegCompResults([]);
                            }}
                            className={`w-full p-3 rounded-lg border text-left transition-all ${
                              compUseCustomDataset
                                ? 'border-violet-500 bg-violet-50/50'
                                : 'border-border hover:border-violet-300 bg-white'
                            }`}
                          >
                            <div className="flex items-center justify-between">
                              <div className="flex items-center gap-2">
                                <FileSpreadsheet className="h-5 w-5 text-violet-500 shrink-0" />
                                <div className="min-w-0">
                                  <p className="text-sm font-semibold truncate">{customDataset.name}</p>
                                  <p className="text-[10px] text-muted-foreground">
                                    {customDataset.nSamples} filas · {customDataset.nFeatures} features · {customDataset.taskType === 'regression' ? 'Regresión' : `${customDataset.nClasses} clases`}
                                  </p>
                                </div>
                              </div>
                              {compUseCustomDataset && (
                                <Badge variant="outline" className="text-[10px] border-violet-400 text-violet-600 bg-violet-50 shrink-0">
                                  Activo
                                </Badge>
                              )}
                            </div>
                          </button>
                          {compUseCustomDataset && (
                            <button
                              onClick={(e) => {
                                e.stopPropagation();
                                setCompUseCustomDataset(false);
                                setCompResults([]);
                                setRegCompResults([]);
                              }}
                              className="absolute -top-1.5 -right-1.5 w-5 h-5 rounded-full bg-violet-500 text-white flex items-center justify-center hover:bg-violet-600 transition-colors shadow-sm"
                              title="Dejar de usar dataset importado"
                            >
                              <X className="h-3 w-3" />
                            </button>
                          )}
                        </div>
                      )}

                      {/* Import CSV button */}
                      <button
                        onClick={() => setShowCSVModal(true)}
                        className={`w-full p-2.5 rounded-lg border-2 border-dashed text-center transition-all mb-3 ${
                          compUseCustomDataset
                            ? 'border-border/50 text-muted-foreground/50 cursor-not-allowed'
                            : 'border-violet-300 hover:border-violet-400 hover:bg-violet-50/50 text-violet-600'
                        }`}
                        disabled={compUseCustomDataset}
                      >
                        <Upload className="h-4 w-4 mx-auto mb-1" />
                        <p className="text-xs font-medium">Importar CSV</p>
                      </button>

                      {/* Built-in datasets */}
                      <p className="text-[10px] text-muted-foreground mb-2 font-medium uppercase tracking-wider">Datasets integrados</p>
                      <div className="grid grid-cols-2 gap-2">
                        {(compTaskType === 'classification' ? DATASET_PRESETS : REGRESSION_DATASET_PRESETS).map((ds) => (
                          <button
                            key={ds.id}
                            onClick={() => {
                              if (compTaskType === 'classification') setCompDataset(ds.id);
                              else setRegCompDataset(ds.id);
                              setCompUseCustomDataset(false);
                              setCompResults([]);
                              setRegCompResults([]);
                            }}
                            className={`p-2 rounded-lg border text-left transition-all ${
                              (compTaskType === 'classification' ? compDataset : regCompDataset) === ds.id && !compUseCustomDataset
                                ? 'border-emerald-500 bg-emerald-50/50'
                                : 'border-border hover:border-emerald-300 bg-white'
                            }`}
                          >
                            <span className="text-lg">{ds.emoji}</span>
                            <p className="text-xs font-semibold mt-1">{ds.name}</p>
                          </button>
                        ))}
                      </div>
                    </CardContent>
                  </Card>

                  {/* Algorithm checkboxes */}
                  <Card>
                    <CardHeader className="pb-2">
                      <CardTitle className="text-sm flex items-center gap-2">
                        <GitCompareArrows className="h-4 w-4 text-emerald-500" />
                        Algoritmos a comparar
                      </CardTitle>
                      <CardDescription className="text-xs">
                        Selecciona al menos 2 algoritmos
                      </CardDescription>
                    </CardHeader>
                    <CardContent>
                      {compTaskType === 'classification' ? (
                        <div className="space-y-3">
                          {ALGORITHMS.map((algo) => (
                            <label
                              key={algo.id}
                              className="flex items-center gap-3 p-2 rounded-lg hover:bg-muted/50 cursor-pointer transition-colors"
                            >
                              <Checkbox
                                checked={compSelectedAlgos.has(algo.id)}
                                onCheckedChange={(checked) => {
                                  setCompSelectedAlgos((prev) => {
                                    const next = new Set(prev);
                                    if (checked) next.add(algo.id);
                                    else next.delete(algo.id);
                                    return next;
                                  });
                                  setCompResults([]);
                                }}
                              />
                              <span className="text-lg">{algo.emoji}</span>
                              <div>
                                <p className="text-sm font-medium">{algo.name}</p>
                                <p className="text-[10px] text-muted-foreground">{algo.shortName}</p>
                              </div>
                            </label>
                          ))}
                        </div>
                      ) : (
                        <div className="space-y-3">
                          {REGRESSION_ALGORITHMS.map((algo) => (
                            <label
                              key={algo.id}
                              className="flex items-center gap-3 p-2 rounded-lg hover:bg-muted/50 cursor-pointer transition-colors"
                            >
                              <Checkbox
                                checked={regCompSelectedAlgos.has(algo.id)}
                                onCheckedChange={(checked) => {
                                  setRegCompSelectedAlgos((prev) => {
                                    const next = new Set(prev);
                                    if (checked) next.add(algo.id);
                                    else next.delete(algo.id);
                                    return next;
                                  });
                                  setRegCompResults([]);
                                }}
                              />
                              <span className="text-lg">{algo.emoji}</span>
                              <div>
                                <p className="text-sm font-medium">{algo.name}</p>
                                <p className="text-[10px] text-muted-foreground">{algo.shortName}</p>
                              </div>
                            </label>
                          ))}
                        </div>
                      )}
                    </CardContent>
                  </Card>

                  {/* Compare button */}
                  <Button
                    onClick={handleCompareTrain}
                    disabled={
                      (compTaskType === 'classification' ? compTraining : regCompTraining) ||
                      (compTaskType === 'classification' ? compSelectedAlgos.size < 2 : regCompSelectedAlgos.size < 2)
                    }
                    className={`w-full h-12 text-base font-semibold text-white shadow-lg ${
                      compTaskType === 'regression'
                        ? 'bg-gradient-to-r from-blue-500 to-indigo-600 hover:from-blue-600 hover:to-indigo-700'
                        : 'bg-gradient-to-r from-emerald-500 to-teal-600 hover:from-emerald-600 hover:to-teal-700'
                    }`}
                    size="lg"
                  >
                    {(compTraining || regCompTraining) ? (
                      <>
                        <Loader2 className="h-5 w-5 mr-2 animate-spin" />
                        {compTrainingAlgo ? `Entrenando ${compTrainingAlgo}...` : 'Comparando...'}
                      </>
                    ) : (
                      <>
                        <GitCompareArrows className="h-5 w-5 mr-2" />
                        Comparar Modelos
                      </>
                    )}
                  </Button>
                </div>

                {/* Results */}
                <div className="lg:col-span-8 space-y-4">
                  {/* ── CLASSIFICATION RESULTS ── */}
                  {compTaskType === 'classification' && compResults.length > 0 && (
                    <>
                      {/* Progress during training */}
                      {compTraining && compTrainingAlgo && (
                        <Card className="border-emerald-200 bg-emerald-50/50">
                          <CardContent className="p-4">
                            <div className="flex items-center gap-3">
                              <Loader2 className="h-5 w-5 animate-spin text-emerald-600" />
                              <div>
                                <p className="text-sm font-medium text-emerald-700">
                                  Entrenando {compTrainingAlgo}...
                                </p>
                                <p className="text-xs text-emerald-600">
                                  {compResults.length} de {compSelectedAlgos.size} modelos completados
                                </p>
                              </div>
                            </div>
                            <Progress
                              value={(compResults.length / compSelectedAlgos.size) * 100}
                              className="mt-3 h-2"
                            />
                          </CardContent>
                        </Card>
                      )}

                      {/* Comparison table */}
                      <Card>
                        <CardHeader className="pb-2">
                          <CardTitle className="text-sm flex items-center gap-2">
                            <BarChart3 className="h-4 w-4 text-emerald-500" />
                            Tabla Comparativa — Clasificación
                          </CardTitle>
                        </CardHeader>
                        <CardContent>
                          <div className="overflow-x-auto">
                            <table className="w-full text-sm">
                              <thead>
                                <tr className="border-b">
                                  <th className="text-left py-2 px-3 font-semibold">Algoritmo</th>
                                  <th className="text-center py-2 px-3 font-semibold">Accuracy</th>
                                  <th className="text-center py-2 px-3 font-semibold">Precision</th>
                                  <th className="text-center py-2 px-3 font-semibold">Recall</th>
                                  <th className="text-center py-2 px-3 font-semibold">F1-Score</th>
                                </tr>
                              </thead>
                              <tbody>
                                {compResults.map((r) => {
                                  const bestAcc = getBestMetricKey('accuracy');
                                  const bestPrec = getBestMetricKey('precision');
                                  const bestRec = getBestMetricKey('recall');
                                  const bestF1 = getBestMetricKey('f1');
                                  return (
                                    <tr key={r.algorithmId} className="border-b last:border-0 hover:bg-muted/30">
                                      <td className="py-2 px-3">
                                        <div className="flex items-center gap-2">
                                          <div
                                            className="w-3 h-3 rounded-full shrink-0"
                                            style={{ backgroundColor: r.color }}
                                          />
                                          <span className="font-medium">{r.algorithmName}</span>
                                        </div>
                                      </td>
                                      <td className={`text-center py-2 px-3 font-mono text-xs ${r.algorithmId === bestAcc ? 'text-green-700 font-bold bg-green-50 rounded' : ''}`}>
                                        {(r.metrics.accuracy * 100).toFixed(1)}%
                                      </td>
                                      <td className={`text-center py-2 px-3 font-mono text-xs ${r.algorithmId === bestPrec ? 'text-green-700 font-bold bg-green-50 rounded' : ''}`}>
                                        {(r.metrics.precision * 100).toFixed(1)}%
                                      </td>
                                      <td className={`text-center py-2 px-3 font-mono text-xs ${r.algorithmId === bestRec ? 'text-green-700 font-bold bg-green-50 rounded' : ''}`}>
                                        {(r.metrics.recall * 100).toFixed(1)}%
                                      </td>
                                      <td className={`text-center py-2 px-3 font-mono text-xs ${r.algorithmId === bestF1 ? 'text-green-700 font-bold bg-green-50 rounded' : ''}`}>
                                        {(r.metrics.f1 * 100).toFixed(1)}%
                                      </td>
                                    </tr>
                                  );
                                })}
                              </tbody>
                            </table>
                          </div>
                          <p className="text-[10px] text-muted-foreground mt-2">
                            🟢 Los valores en verde son los mejores de cada métrica
                          </p>
                        </CardContent>
                      </Card>

                      {/* Mini decision boundaries */}
                      <Card>
                        <CardHeader className="pb-2">
                          <CardTitle className="text-sm">Fronteras de Decisión Comparadas</CardTitle>
                        </CardHeader>
                        <CardContent>
                          <div className="grid grid-cols-2 sm:grid-cols-3 gap-4">
                            {compResults.map((r) => (
                              <div key={r.algorithmId} className="flex flex-col items-center gap-2">
                                <div className="flex items-center gap-1.5">
                                  <div
                                    className="w-2.5 h-2.5 rounded-full"
                                    style={{ backgroundColor: r.color }}
                                  />
                                  <span className="text-xs font-semibold">{r.algorithmName}</span>
                                </div>
                                {r.boundaryData ? (
                                  <MiniDecisionBoundaryCanvas data={r.boundaryData} width={200} height={200} />
                                ) : (
                                  <div className="w-[200px] h-[200px] bg-muted/30 rounded-lg border border-dashed flex items-center justify-center">
                                    <p className="text-[10px] text-muted-foreground">No disponible</p>
                                  </div>
                                )}
                                <p className="text-[10px] text-muted-foreground">
                                  F1: {(r.metrics.f1 * 100).toFixed(1)}%
                                </p>
                              </div>
                            ))}
                          </div>
                        </CardContent>
                      </Card>

                      {/* F1 Bar Chart */}
                      <Card>
                        <CardContent className="p-4">
                          <F1BarChart results={compResults} />
                        </CardContent>
                      </Card>
                    </>
                  )}

                  {/* ── REGRESSION RESULTS ── */}
                  {compTaskType === 'regression' && regCompResults.length > 0 && (
                    <>
                      {/* Progress during training */}
                      {regCompTraining && compTrainingAlgo && (
                        <Card className="border-blue-200 bg-blue-50/50">
                          <CardContent className="p-4">
                            <div className="flex items-center gap-3">
                              <Loader2 className="h-5 w-5 animate-spin text-blue-600" />
                              <div>
                                <p className="text-sm font-medium text-blue-700">
                                  Entrenando {compTrainingAlgo}...
                                </p>
                                <p className="text-xs text-blue-600">
                                  {regCompResults.length} de {regCompSelectedAlgos.size} modelos completados
                                </p>
                              </div>
                            </div>
                            <Progress
                              value={(regCompResults.length / regCompSelectedAlgos.size) * 100}
                              className="mt-3 h-2 [&>div]:bg-blue-500"
                            />
                          </CardContent>
                        </Card>
                      )}

                      {/* Regression comparison table */}
                      <Card>
                        <CardHeader className="pb-2">
                          <CardTitle className="text-sm flex items-center gap-2">
                            <BarChart3 className="h-4 w-4 text-blue-500" />
                            Tabla Comparativa — Regresión
                          </CardTitle>
                        </CardHeader>
                        <CardContent>
                          <div className="overflow-x-auto">
                            <table className="w-full text-sm">
                              <thead>
                                <tr className="border-b">
                                  <th className="text-left py-2 px-3 font-semibold">Algoritmo</th>
                                  <th className="text-center py-2 px-3 font-semibold">R²</th>
                                  <th className="text-center py-2 px-3 font-semibold">MSE</th>
                                  <th className="text-center py-2 px-3 font-semibold">RMSE</th>
                                  <th className="text-center py-2 px-3 font-semibold">MAE</th>
                                </tr>
                              </thead>
                              <tbody>
                                {regCompResults.map((r) => {
                                  const bestR2 = regCompResults.reduce((a, b) => a.metrics.r2 >= b.metrics.r2 ? a : b).algorithmId;
                                  const bestMSE = regCompResults.reduce((a, b) => a.metrics.mse <= b.metrics.mse ? a : b).algorithmId;
                                  const bestRMSE = regCompResults.reduce((a, b) => a.metrics.rmse <= b.metrics.rmse ? a : b).algorithmId;
                                  const bestMAE = regCompResults.reduce((a, b) => a.metrics.mae <= b.metrics.mae ? a : b).algorithmId;
                                  return (
                                    <tr key={r.algorithmId} className="border-b last:border-0 hover:bg-muted/30">
                                      <td className="py-2 px-3">
                                        <div className="flex items-center gap-2">
                                          <div
                                            className="w-3 h-3 rounded-full shrink-0"
                                            style={{ backgroundColor: r.color }}
                                          />
                                          <span className="font-medium">{r.algorithmName}</span>
                                        </div>
                                      </td>
                                      <td className={`text-center py-2 px-3 font-mono text-xs ${r.algorithmId === bestR2 ? 'text-green-700 font-bold bg-green-50 rounded' : ''}`}>
                                        {r.metrics.r2.toFixed(4)}
                                      </td>
                                      <td className={`text-center py-2 px-3 font-mono text-xs ${r.algorithmId === bestMSE ? 'text-green-700 font-bold bg-green-50 rounded' : ''}`}>
                                        {r.metrics.mse.toFixed(4)}
                                      </td>
                                      <td className={`text-center py-2 px-3 font-mono text-xs ${r.algorithmId === bestRMSE ? 'text-green-700 font-bold bg-green-50 rounded' : ''}`}>
                                        {r.metrics.rmse.toFixed(4)}
                                      </td>
                                      <td className={`text-center py-2 px-3 font-mono text-xs ${r.algorithmId === bestMAE ? 'text-green-700 font-bold bg-green-50 rounded' : ''}`}>
                                        {r.metrics.mae.toFixed(4)}
                                      </td>
                                    </tr>
                                  );
                                })}
                              </tbody>
                            </table>
                          </div>
                          <p className="text-[10px] text-muted-foreground mt-2">
                            🟢 R² más alto = mejor · MSE/RMSE/MAE más bajo = mejor
                          </p>
                        </CardContent>
                      </Card>

                      {/* R² Bar Chart */}
                      <Card>
                        <CardContent className="p-4">
                          <div className="flex flex-col items-center">
                            <p className="text-sm font-semibold mb-2 text-foreground">Comparación de R² (Coeficiente de Determinación)</p>
                            {(() => {
                              const maxR2 = Math.max(...regCompResults.map((r) => Math.abs(r.metrics.r2)), 0.01);
                              const barHeight = 32;
                              const chartWidth = 300;
                              const labelWidth = 150;
                              const totalWidth = labelWidth + chartWidth + 80;
                              const totalHeight = regCompResults.length * (barHeight + 12) + 20;
                              return (
                                <svg width={totalWidth} height={totalHeight}>
                                  {regCompResults.map((item, i) => {
                                    const isBest = item.metrics.r2 === Math.max(...regCompResults.map((r) => r.metrics.r2));
                                    const barW = maxR2 > 0 ? (Math.abs(item.metrics.r2) / maxR2) * chartWidth : 0;
                                    const y = i * (barHeight + 12) + 5;
                                    return (
                                      <g key={item.algorithmId}>
                                        <text x={labelWidth - 5} y={y + barHeight / 2 + 4} textAnchor="end" className="text-xs fill-foreground font-medium" fontSize={11}>
                                          {item.algorithmName.length > 18 ? item.algorithmName.substring(0, 16) + '…' : item.algorithmName}
                                        </text>
                                        <rect x={labelWidth} y={y} width={barW} height={barHeight} fill={item.color} rx={6} opacity={isBest ? 1 : 0.6} />
                                        <text x={labelWidth + barW + 8} y={y + barHeight / 2 + 5} className="text-xs fill-foreground font-bold" fontSize={12}>
                                          {item.metrics.r2.toFixed(4)}{isBest ? ' ★' : ''}
                                        </text>
                                      </g>
                                    );
                                  })}
                                </svg>
                              );
                            })()}
                          </div>
                        </CardContent>
                      </Card>
                    </>
                  )}

                  {/* Empty state */}
                  {((compTaskType === 'classification' && compResults.length === 0) ||
                    (compTaskType === 'regression' && regCompResults.length === 0)) && !(compTraining || regCompTraining) && (
                    <Card className="border-dashed">
                      <CardContent className="p-8">
                        <div className="text-center">
                          <GitCompareArrows className={`h-16 w-16 mx-auto mb-4 ${compTaskType === 'regression' ? 'text-blue-300' : 'text-emerald-300'}`} />
                          <h3 className="text-lg font-semibold text-foreground mb-2">
                            Compara modelos lado a lado
                          </h3>
                          <p className="text-sm text-muted-foreground max-w-md mx-auto">
                            {compTaskType === 'classification'
                              ? 'Selecciona un dataset y marca los algoritmos de clasificación que deseas comparar. Puedes usar un dataset importado (CSV) o uno de los presets.'
                              : 'Selecciona un dataset y marca los algoritmos de regresión que deseas comparar. Puedes usar un dataset importado (CSV) o uno de los presets.'}
                          </p>
                          {!customDataset && (
                            <div className="mt-4">
                              <Button
                                variant="outline"
                                size="sm"
                                onClick={() => setShowCSVModal(true)}
                                className="gap-1.5"
                              >
                                <Upload className="h-4 w-4" />
                                Importar CSV para comparar
                              </Button>
                            </div>
                          )}
                        </div>
                      </CardContent>
                    </Card>
                  )}
                </div>
              </div>
            </TabsContent>

            {/* ═══════════════════════════════════════════════════════
                TAB: TALLERES
                ═══════════════════════════════════════════════════════ */}
            <TabsContent value="workshops" className="mt-6">
              <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
                {/* Left: Workshop list / info */}
                <div className="lg:col-span-4 space-y-4">
                  {/* Workshop list */}
                  {!selectedWorkshop ? (
                    <div className="space-y-3">
                      <h2 className="text-sm font-semibold text-muted-foreground uppercase tracking-wider">
                        Talleres Disponibles
                      </h2>
                      {WORKSHOPS.map((ws) => (
                        <Card
                          key={ws.id}
                          className="cursor-pointer hover:border-emerald-400 hover:shadow-md transition-all"
                          onClick={() => handleSelectWorkshop(ws)}
                        >
                          <CardContent className="p-4">
                            <div className="flex items-start gap-3">
                              <span className="text-3xl">{ws.emoji}</span>
                              <div className="flex-1 min-w-0">
                                <div className="flex items-center gap-2 mb-1">
                                  <h3 className="text-sm font-bold text-foreground">{ws.title}</h3>
                                </div>
                                <div className="flex items-center gap-2 mb-2">
                                  <Badge
                                    variant="outline"
                                    className={`text-[10px] px-1.5 py-0 ${diffColors[ws.difficulty] || ''}`}
                                  >
                                    {ws.difficulty}
                                  </Badge>
                                  <span className="text-[10px] text-muted-foreground flex items-center gap-1">
                                    <Clock className="h-3 w-3" />
                                    {ws.duration}
                                  </span>
                                  <span className="text-[10px] text-muted-foreground">
                                    {ws.steps.length} pasos
                                  </span>
                                </div>
                                <p className="text-xs text-muted-foreground line-clamp-2">
                                  {ws.description}
                                </p>
                              </div>
                              <ChevronRight className="h-5 w-5 text-muted-foreground shrink-0 mt-1" />
                            </div>
                          </CardContent>
                        </Card>
                      ))}
                    </div>
                  ) : (
                    <div className="space-y-4">
                      {/* Back button */}
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={() => {
                          setSelectedWorkshop(null);
                          setWsTrainResult(null);
                          setWsBoundaryData(null);
                        }}
                        className="text-xs"
                      >
                        ← Volver a talleres
                      </Button>

                      {/* Workshop header */}
                      <Card className="overflow-hidden">
                        <div className="h-2 bg-gradient-to-r from-emerald-500 to-teal-500" />
                        <CardHeader className="pb-2">
                          <div className="flex items-start gap-3">
                            <span className="text-3xl">{selectedWorkshop.emoji}</span>
                            <div>
                              <CardTitle className="text-lg">{selectedWorkshop.title}</CardTitle>
                              <div className="flex items-center gap-2 mt-1">
                                <Badge
                                  variant="outline"
                                  className={`text-[10px] px-1.5 py-0 ${diffColors[selectedWorkshop.difficulty] || ''}`}
                                >
                                  {selectedWorkshop.difficulty}
                                </Badge>
                                <span className="text-[10px] text-muted-foreground flex items-center gap-1">
                                  <Clock className="h-3 w-3" />
                                  {selectedWorkshop.duration}
                                </span>
                              </div>
                            </div>
                          </div>
                        </CardHeader>
                        <CardContent>
                          <p className="text-sm text-muted-foreground mb-3">
                            {selectedWorkshop.description}
                          </p>
                          <div>
                            <h4 className="text-xs font-semibold mb-2 flex items-center gap-1">
                              <Target className="h-3.5 w-3.5 text-emerald-500" />
                              Objetivos de aprendizaje
                            </h4>
                            <ul className="space-y-1">
                              {selectedWorkshop.learningObjectives.map((obj, i) => (
                                <li key={i} className="text-xs text-muted-foreground flex items-start gap-1.5">
                                  <CheckCircle2 className="h-3 w-3 text-emerald-500 shrink-0 mt-0.5" />
                                  {obj}
                                </li>
                              ))}
                            </ul>
                          </div>

                          {/* Progress bar */}
                          <div className="mt-4">
                            <div className="flex items-center justify-between mb-1">
                              <span className="text-xs font-medium">Progreso</span>
                              <span className="text-xs text-muted-foreground">
                                {completedSteps.size}/{selectedWorkshop.steps.length} pasos
                              </span>
                            </div>
                            <Progress
                              value={(completedSteps.size / selectedWorkshop.steps.length) * 100}
                              className="h-2"
                            />
                          </div>
                        </CardContent>
                      </Card>

                      {/* Steps list */}
                      <Card>
                        <CardHeader className="pb-2">
                          <CardTitle className="text-sm">Pasos del Taller</CardTitle>
                        </CardHeader>
                        <CardContent>
                          <ScrollArea className="max-h-[400px]">
                            <div className="space-y-1">
                              {selectedWorkshop.steps.map((step, idx) => (
                                <button
                                  key={idx}
                                  onClick={() => {
                                    setCurrentStepIdx(idx);
                                    setShowHint(false);
                                  }}
                                  className={`w-full text-left p-2.5 rounded-lg transition-all flex items-center gap-2 ${
                                    currentStepIdx === idx
                                      ? 'bg-emerald-50 border border-emerald-200'
                                      : 'hover:bg-muted/50'
                                  }`}
                                >
                                  <div className={`w-6 h-6 rounded-full flex items-center justify-center shrink-0 text-[10px] font-bold ${
                                    completedSteps.has(idx)
                                      ? 'bg-emerald-500 text-white'
                                      : currentStepIdx === idx
                                        ? 'bg-emerald-100 text-emerald-700 border border-emerald-300'
                                        : 'bg-muted text-muted-foreground'
                                  }`}>
                                    {completedSteps.has(idx) ? (
                                      <CheckCircle2 className="h-3.5 w-3.5" />
                                    ) : (
                                      idx + 1
                                    )}
                                  </div>
                                  <span className={`text-xs ${
                                    currentStepIdx === idx
                                      ? 'font-semibold text-emerald-700'
                                      : completedSteps.has(idx)
                                        ? 'text-muted-foreground line-through'
                                        : 'text-foreground'
                                  }`}>
                                    Paso {idx + 1}
                                  </span>
                                                {idx < selectedWorkshop.steps.length - 1 && (
                                  <ArrowRight className="h-3 w-3 text-muted-foreground ml-auto" />
                                )}
                                </button>
                              ))}
                            </div>
                          </ScrollArea>
                        </CardContent>
                      </Card>
                    </div>
                  )}
                </div>

                {/* Right: Step detail */}
                <div className="lg:col-span-8 space-y-4">
                  {selectedWorkshop ? (
                    <>
                      {(() => {
                        const step = selectedWorkshop.steps[currentStepIdx];
                        if (!step) return null;
                        const algo = ALGORITHMS.find((a) => a.id === step.algorithm);
                        const ds = DATASET_PRESETS.find((d) => d.id === step.dataset);
                        return (
                          <>
                            {/* Step instruction */}
                            <Card className="overflow-hidden">
                              <div className="h-1.5" style={{ backgroundColor: algo?.color || '#10b981' }} />
                              <CardHeader className="pb-2">
                                <div className="flex items-center justify-between">
                                  <CardTitle className="text-base flex items-center gap-2">
                                    <div className={`w-7 h-7 rounded-full flex items-center justify-center text-xs font-bold ${
                                      completedSteps.has(currentStepIdx)
                                        ? 'bg-emerald-500 text-white'
                                        : 'bg-emerald-100 text-emerald-700'
                                    }`}>
                                      {completedSteps.has(currentStepIdx) ? (
                                        <CheckCircle2 className="h-4 w-4" />
                                      ) : (
                                        currentStepIdx + 1
                                      )}
                                    </div>
                                    Paso {currentStepIdx + 1} de {selectedWorkshop.steps.length}
                                  </CardTitle>
                                  <div className="flex items-center gap-1.5">
                                    <Badge variant="outline" className="text-[10px]">
                                      {algo?.emoji} {algo?.name}
                                    </Badge>
                                    <Badge variant="secondary" className="text-[10px]">
                                      {ds?.emoji} {ds?.name}
                                    </Badge>
                                  </div>
                                </div>
                              </CardHeader>
                              <CardContent className="space-y-4">
                                <p className="text-sm text-foreground leading-relaxed">
                                  {step.instruction}
                                </p>

                                {/* Action buttons */}
                                <div className="flex flex-wrap gap-2">
                                  <Button
                                    onClick={() => handleConfigureStep(step)}
                                    className="bg-emerald-500 hover:bg-emerald-600 text-white"
                                    size="sm"
                                  >
                                    <Zap className="h-4 w-4 mr-1" />
                                    Configurar Paso
                                  </Button>
                                  <Button
                                    onClick={() => handleWorkshopTrain(step)}
                                    disabled={wsTraining}
                                    className="bg-gradient-to-r from-emerald-500 to-teal-600 hover:from-emerald-600 hover:to-teal-700 text-white"
                                    size="sm"
                                  >
                                    {wsTraining ? (
                                      <>
                                        <Loader2 className="h-4 w-4 mr-1 animate-spin" />
                                        Entrenando...
                                      </>
                                    ) : (
                                      <>
                                        <Play className="h-4 w-4 mr-1" />
                                        Entrenar
                                      </>
                                    )}
                                  </Button>
                                  {!completedSteps.has(currentStepIdx) && (
                                    <Button
                                      onClick={handleCompleteStep}
                                      variant="outline"
                                      size="sm"
                                      className="border-emerald-300 text-emerald-600 hover:bg-emerald-50"
                                    >
                                      <CheckCircle2 className="h-4 w-4 mr-1" />
                                      Marcar completado
                                    </Button>
                                  )}
                                </div>

                                {/* Navigation */}
                                <div className="flex items-center justify-between pt-2">
                                  <Button
                                    variant="ghost"
                                    size="sm"
                                    disabled={currentStepIdx === 0}
                                    onClick={() => {
                                      setCurrentStepIdx((prev) => Math.max(0, prev - 1));
                                      setShowHint(false);
                                      setWsTrainResult(null);
                                      setWsBoundaryData(null);
                                    }}
                                  >
                                    ← Anterior
                                  </Button>
                                  <Button
                                    variant="ghost"
                                    size="sm"
                                    disabled={currentStepIdx >= selectedWorkshop.steps.length - 1}
                                    onClick={() => {
                                      setCurrentStepIdx((prev) => Math.min(selectedWorkshop.steps.length - 1, prev + 1));
                                      setShowHint(false);
                                      setWsTrainResult(null);
                                      setWsBoundaryData(null);
                                    }}
                                  >
                                    Siguiente →
                                  </Button>
                                </div>

                                <Separator />

                                {/* Question */}
                                <div className="bg-amber-50 border border-amber-200 rounded-lg p-4">
                                  <h4 className="text-sm font-semibold text-amber-800 flex items-center gap-1.5 mb-2">
                                    <HelpCircle className="h-4 w-4" />
                                    Pregunta
                                  </h4>
                                  <p className="text-sm text-amber-900">{step.question}</p>
                                </div>

                                {/* Hint */}
                                <div>
                                  <Button
                                    variant="ghost"
                                    size="sm"
                                    onClick={() => setShowHint(!showHint)}
                                    className="text-xs"
                                  >
                                    <Lightbulb className="h-4 w-4 mr-1 text-yellow-500" />
                                    {showHint ? 'Ocultar pista' : 'Mostrar pista'}
                                  </Button>
                                  {showHint && (
                                    <div className="mt-2 bg-yellow-50 border border-yellow-200 rounded-lg p-3">
                                      <p className="text-sm text-yellow-800">{step.hint}</p>
                                    </div>
                                  )}
                                </div>

                                {/* Workshop train results */}
                                {wsTrainResult?.success && wsTrainResult.metrics && (
                                  <>
                                    <Separator />
                                    <div className="space-y-3">
                                      <h4 className="text-xs font-semibold text-muted-foreground uppercase">
                                        Resultados del entrenamiento
                                      </h4>
                                      <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
                                        <MetricCard
                                          label="Accuracy"
                                          value={wsTrainResult.metrics.accuracy}
                                          icon={<Target className="h-4 w-4 text-emerald-500" />}
                                          color="bg-emerald-500"
                                        />
                                        <MetricCard
                                          label="Precision"
                                          value={wsTrainResult.metrics.precision}
                                          icon={<BarChart3 className="h-4 w-4 text-teal-500" />}
                                          color="bg-teal-500"
                                        />
                                        <MetricCard
                                          label="Recall"
                                          value={wsTrainResult.metrics.recall}
                                          icon={<TrendingUp className="h-4 w-4 text-amber-500" />}
                                          color="bg-amber-500"
                                        />
                                        <MetricCard
                                          label="F1-Score"
                                          value={wsTrainResult.metrics.f1}
                                          icon={<Zap className="h-4 w-4 text-violet-500" />}
                                          color="bg-violet-500"
                                        />
                                      </div>
                                      {wsBoundaryData && (
                                        <MiniDecisionBoundaryCanvas data={wsBoundaryData} width={300} height={300} />
                                      )}
                                    </div>
                                  </>
                                )}
                                {wsTrainResult && !wsTrainResult.success && (
                                  <Card className="border-red-200 bg-red-50">
                                    <CardContent className="p-3">
                                      <div className="flex items-center gap-2 text-red-700">
                                        <XCircle className="h-4 w-4" />
                                        <p className="text-xs">{wsTrainResult.error}</p>
                                      </div>
                                    </CardContent>
                                  </Card>
                                )}
                              </CardContent>
                            </Card>

                            {/* Final Challenge */}
                            {completedSteps.size === selectedWorkshop.steps.length && (
                              <Card className="border-emerald-300 bg-gradient-to-br from-emerald-50 to-teal-50">
                                <CardHeader className="pb-2">
                                  <CardTitle className="text-base flex items-center gap-2 text-emerald-700">
                                    <Trophy className="h-5 w-5" />
                                    ¡Desafío Final!
                                  </CardTitle>
                                </CardHeader>
                                <CardContent>
                                  <p className="text-sm text-emerald-800 mb-3">
                                    {selectedWorkshop.finalChallenge}
                                  </p>
                                  <Button
                                    onClick={() => setMainTab('lab')}
                                    className="bg-emerald-500 hover:bg-emerald-600 text-white"
                                    size="sm"
                                  >
                                    <Flame className="h-4 w-4 mr-1" />
                                    Ir al Laboratorio
                                  </Button>
                                </CardContent>
                              </Card>
                            )}
                          </>
                        );
                      })()}
                    </>
                  ) : (
                    <Card className="border-dashed">
                      <CardContent className="p-8">
                        <div className="text-center">
                          <GraduationCap className="h-16 w-16 mx-auto mb-4 text-emerald-300" />
                          <h3 className="text-lg font-semibold text-foreground mb-2">
                            Talleres Guiados
                          </h3>
                          <p className="text-sm text-muted-foreground max-w-md mx-auto">
                            Selecciona un taller de la lista para comenzar tu aprendizaje guiado.
                            Cada taller te llevará paso a paso por conceptos clave de ML con
                            instrucciones, preguntas y experimentos prácticos.
                          </p>
                          <div className="mt-4 grid grid-cols-3 gap-3 max-w-sm mx-auto">
                            <div className="text-center">
                              <p className="text-2xl font-bold text-emerald-600">{WORKSHOPS.length}</p>
                              <p className="text-[10px] text-muted-foreground">Talleres</p>
                            </div>
                            <div className="text-center">
                              <p className="text-2xl font-bold text-emerald-600">
                                {WORKSHOPS.reduce((a, w) => a + w.steps.length, 0)}
                              </p>
                              <p className="text-[10px] text-muted-foreground">Pasos</p>
                            </div>
                            <div className="text-center">
                              <p className="text-2xl font-bold text-emerald-600">3</p>
                              <p className="text-[10px] text-muted-foreground">Niveles</p>
                            </div>
                          </div>
                        </div>
                      </CardContent>
                    </Card>
                  )}
                </div>
              </div>
            </TabsContent>
          </Tabs>
        </main>

        {/* Footer */}
        <footer className="mt-auto border-t bg-white/50">
          <div className="max-w-7xl mx-auto px-4 py-4 text-center text-xs text-muted-foreground">
            ML HyperLab — Laboratorio Interactivo de Machine Learning — Todo corre en tu navegador 🚀
          </div>
        </footer>
      </div>

      {/* CSV Import Modal */}
      {showCSVModal && (
        <div className="fixed inset-0 z-[100] flex items-center justify-center bg-black/50 p-4">
          <div className="bg-white rounded-2xl shadow-2xl w-full max-w-2xl max-h-[90vh] flex flex-col overflow-hidden">
            {/* Modal Header */}
            <div className="flex items-center justify-between p-4 border-b shrink-0">
              <div className="flex items-center gap-2">
                <FileSpreadsheet className="h-5 w-5 text-emerald-600" />
                <h2 className="text-lg font-semibold">Importar Dataset CSV</h2>
              </div>
              <Button
                variant="ghost"
                size="sm"
                onClick={() => {
                  setShowCSVModal(false);
                  setCsvImportStep('upload');
                  setCsvParseProgress(null);
                }}
                className="h-8 w-8 p-0"
              >
                <X className="h-4 w-4" />
              </Button>
            </div>

            {/* Step indicators */}
            <div className="flex items-center gap-2 px-4 py-2 border-b bg-slate-50 shrink-0">
              {(['upload', 'configure', 'preview'] as const).map((step, idx) => {
                const stepLabels: Record<string, string> = {
                  upload: 'Subir archivo',
                  configure: 'Configurar',
                  preview: 'Vista previa',
                };
                const isActive = csvImportStep === step;
                const isCompleted = csvImportStep === 'configure' && idx === 0
                  || csvImportStep === 'preview' && idx < 2;
                return (
                  <div key={step} className="flex items-center gap-2">
                    {idx > 0 && <ChevronRight className="h-3 w-3 text-muted-foreground" />}
                    <div className={`flex items-center gap-1.5 text-xs font-medium ${
                      isActive ? 'text-emerald-700' : isCompleted ? 'text-emerald-500' : 'text-muted-foreground'
                    }`}>
                      <div className={`w-5 h-5 rounded-full flex items-center justify-center text-[10px] font-bold ${
                        isActive ? 'bg-emerald-500 text-white' : isCompleted ? 'bg-emerald-100 text-emerald-600' : 'bg-muted text-muted-foreground'
                      }`}>
                        {isCompleted ? <CheckCircle2 className="h-3 w-3" /> : idx + 1}
                      </div>
                      {stepLabels[step]}
                    </div>
                  </div>
                );
              })}
            </div>

            {/* Modal Content */}
            <div className="flex-1 overflow-y-auto p-4">
              {/* UPLOAD STEP */}
              {csvImportStep === 'upload' && (
                <div className="space-y-4">
                  <div
                    onDragOver={(e) => { e.preventDefault(); setCsvDragOver(true); }}
                    onDragLeave={() => setCsvDragOver(false)}
                    onDrop={handleCSVDrop}
                    className={`border-2 border-dashed rounded-xl p-8 text-center transition-all ${
                      csvDragOver
                        ? 'border-emerald-500 bg-emerald-50/50'
                        : 'border-gray-300 hover:border-emerald-400 hover:bg-gray-50/50'
                    }`}
                  >
                    <Upload className={`h-12 w-12 mx-auto mb-3 ${csvDragOver ? 'text-emerald-500' : 'text-gray-400'}`} />
                    <p className="text-sm font-semibold text-foreground mb-1">
                      Arrastra tu archivo CSV aquí
                    </p>
                    <p className="text-xs text-muted-foreground mb-3">
                      o haz clic para seleccionar un archivo
                    </p>
                    <label className="inline-block">
                      <input
                        type="file"
                        accept=".csv,.txt,.tsv"
                        className="hidden"
                        onChange={(e) => {
                          const file = e.target.files?.[0];
                          if (file) handleCSVFile(file);
                        }}
                      />
                      <span className="inline-flex items-center gap-1.5 px-4 py-2 rounded-lg bg-emerald-500 text-white text-sm font-medium cursor-pointer hover:bg-emerald-600 transition-colors">
                        <FileSpreadsheet className="h-4 w-4" />
                        Seleccionar archivo
                      </span>
                    </label>
                    <p className="text-[10px] text-muted-foreground mt-3">
                      Formatos soportados: CSV, TSV, TXT · Máximo 10,000 filas
                    </p>
                  </div>

                  {/* Parsing Progress */}
                  {csvParseProgress && (
                    <div className="space-y-2">
                      <div className="flex items-center justify-between text-xs">
                        <span className="text-muted-foreground">{csvParseProgress.message}</span>
                        <span className="font-mono text-emerald-600">{csvParseProgress.progress}%</span>
                      </div>
                      <Progress
                        value={csvParseProgress.progress}
                        className={`h-2 ${csvParseProgress.stage === 'error' ? '[&>div]:bg-red-500' : '[&>div]:bg-emerald-500'}`}
                      />
                      {csvParseProgress.stage === 'error' && (
                        <div className="flex items-center gap-2 text-red-600 text-sm mt-2">
                          <XCircle className="h-4 w-4" />
                          <p>{csvParseProgress.message}</p>
                        </div>
                      )}
                    </div>
                  )}
                </div>
              )}

              {/* CONFIGURE STEP */}
              {csvImportStep === 'configure' && csvColumns.length > 0 && (
                <div className="space-y-4">
                  {/* File stats */}
                  <div className="grid grid-cols-3 gap-3">
                    <div className="bg-emerald-50 rounded-lg p-3 text-center">
                      <Database className="h-4 w-4 mx-auto text-emerald-600 mb-1" />
                      <p className="text-lg font-bold text-emerald-700">{csvRows.length.toLocaleString()}</p>
                      <p className="text-[10px] text-emerald-600">Filas</p>
                    </div>
                    <div className="bg-teal-50 rounded-lg p-3 text-center">
                      <Columns className="h-4 w-4 mx-auto text-teal-600 mb-1" />
                      <p className="text-lg font-bold text-teal-700">{csvColumns.length}</p>
                      <p className="text-[10px] text-teal-600">Columnas</p>
                    </div>
                    <div className="bg-amber-50 rounded-lg p-3 text-center">
                      <FileSpreadsheet className="h-4 w-4 mx-auto text-amber-600 mb-1" />
                      <p className="text-lg font-bold text-amber-700">{csvFileName}</p>
                      <p className="text-[10px] text-amber-600 truncate">Archivo</p>
                    </div>
                  </div>

                  {/* Target column selector */}
                  <div className="space-y-2">
                    <label className="text-sm font-semibold flex items-center gap-1.5">
                      <Target className="h-4 w-4 text-emerald-500" />
                      Columna objetivo (Target)
                    </label>
                    <p className="text-xs text-muted-foreground">
                      Selecciona la columna que quieres predecir. Las demás columnas se usarán como features.
                    </p>
                    <Select value={csvTargetColumn} onValueChange={setCsvTargetColumn}>
                      <SelectTrigger className="w-full">
                        <SelectValue placeholder="Selecciona la columna objetivo" />
                      </SelectTrigger>
                      <SelectContent>
                        {csvColumns.map((col) => (
                          <SelectItem key={col} value={col}>
                            {col}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>

                  {/* Warning for too many classes */}
                  {csvTargetColumn && (() => {
                    const targetIdx = csvColumns.indexOf(csvTargetColumn);
                    const targetTypeInfo = csvColumnTypes[targetIdx];
                    if (targetTypeInfo && targetTypeInfo.uniqueValues > 10) {
                      return (
                        <div className="flex items-start gap-2 p-3 bg-amber-50 border border-amber-200 rounded-lg">
                          <AlertTriangle className="h-4 w-4 text-amber-500 shrink-0 mt-0.5" />
                          <div>
                            <p className="text-xs font-semibold text-amber-800">Muchas clases detectadas</p>
                            <p className="text-xs text-amber-700">
                              La columna objetivo tiene {targetTypeInfo.uniqueValues} valores únicos.
                              Para clasificación, se recomienda usar una columna con 2-10 clases.
                            </p>
                          </div>
                        </div>
                      );
                    }
                    return null;
                  })()}

                  {/* Task Type Override Selector */}
                  {csvTargetColumn && (() => {
                    const autoDetected = detectTaskType(csvTargetColumn, csvColumnTypes, csvColumns, csvRows);
                    const targetIdx = csvColumns.indexOf(csvTargetColumn);
                    const targetTypeInfo = csvColumnTypes[targetIdx];
                    const uniqueVals = targetTypeInfo?.uniqueValues || 0;
                    const isAutoClassification = autoDetected === 'classification';
                    const isAutoRegression = autoDetected === 'regression';
                    const currentEffective = csvTaskTypeOverride === 'auto' ? autoDetected : csvTaskTypeOverride;
                    const isOverride = csvTaskTypeOverride !== 'auto';

                    return (
                      <div className="space-y-2">
                        <label className="text-sm font-semibold flex items-center gap-1.5">
                          <Brain className="h-4 w-4 text-emerald-500" />
                          Tipo de tarea
                        </label>
                        <p className="text-xs text-muted-foreground">
                          Selecciona si tu problema es de clasificación (predecir categorías) o regresión (predecir valores continuos).
                        </p>

                        {/* Auto-detection result */}
                        <div className={`flex items-center gap-2 p-2.5 rounded-lg border text-xs ${
                          isAutoClassification
                            ? 'bg-emerald-50 border-emerald-200 text-emerald-800'
                            : 'bg-blue-50 border-blue-200 text-blue-800'
                        }`}>
                          <Lightbulb className="h-3.5 w-3.5 shrink-0" />
                          <span>
                            Auto-detección: <strong>{isAutoClassification ? 'Clasificación' : 'Regresión'}</strong>
                            {' '}({uniqueVals} valores únicos
                            {targetTypeInfo && targetTypeInfo.type === 'numeric' && !csvRows.every((r) => Number.isInteger(Number(r[targetIdx]))) 
                              ? ', valores decimales' 
                              : ''})
                          </span>
                        </div>

                        {/* Task type selector */}
                        <div className="grid grid-cols-3 gap-2">
                          <button
                            onClick={() => setCsvTaskTypeOverride('auto')}
                            className={`flex flex-col items-center gap-1 p-2.5 rounded-lg border text-xs font-medium transition-all ${
                              csvTaskTypeOverride === 'auto'
                                ? 'border-emerald-400 bg-emerald-50 text-emerald-700 ring-1 ring-emerald-200'
                                : 'border-gray-200 text-muted-foreground hover:bg-muted/50'
                            }`}
                          >
                            <Zap className="h-4 w-4" />
                            <span>Auto</span>
                            {csvTaskTypeOverride === 'auto' && (
                              <span className="text-[9px] text-emerald-600">({isAutoClassification ? 'Clasif.' : 'Regresión'})</span>
                            )}
                          </button>
                          <button
                            onClick={() => setCsvTaskTypeOverride('classification')}
                            className={`flex flex-col items-center gap-1 p-2.5 rounded-lg border text-xs font-medium transition-all ${
                              csvTaskTypeOverride === 'classification'
                                ? 'border-emerald-400 bg-emerald-50 text-emerald-700 ring-1 ring-emerald-200'
                                : currentEffective === 'classification' && csvTaskTypeOverride === 'auto'
                                  ? 'border-emerald-200 bg-emerald-50/30 text-emerald-600'
                                  : 'border-gray-200 text-muted-foreground hover:bg-muted/50'
                            }`}
                          >
                            <Target className="h-4 w-4" />
                            <span>Clasificación</span>
                          </button>
                          <button
                            onClick={() => setCsvTaskTypeOverride('regression')}
                            className={`flex flex-col items-center gap-1 p-2.5 rounded-lg border text-xs font-medium transition-all ${
                              csvTaskTypeOverride === 'regression'
                                ? 'border-blue-400 bg-blue-50 text-blue-700 ring-1 ring-blue-200'
                                : currentEffective === 'regression' && csvTaskTypeOverride === 'auto'
                                  ? 'border-blue-200 bg-blue-50/30 text-blue-600'
                                  : 'border-gray-200 text-muted-foreground hover:bg-muted/50'
                            }`}
                          >
                            <TrendingUp className="h-4 w-4" />
                            <span>Regresión</span>
                          </button>
                        </div>

                        {/* Override warning */}
                        {isOverride && (
                          <div className={`flex items-start gap-2 p-2.5 rounded-lg border text-xs ${
                            csvTaskTypeOverride === 'classification' && isAutoRegression
                              ? 'bg-amber-50 border-amber-200'
                              : csvTaskTypeOverride === 'regression' && isAutoClassification
                                ? 'bg-amber-50 border-amber-200'
                                : 'bg-emerald-50 border-emerald-200'
                          }`}>
                            {(csvTaskTypeOverride === 'classification' && isAutoRegression) || 
                             (csvTaskTypeOverride === 'regression' && isAutoClassification) ? (
                              <>
                                <AlertTriangle className="h-3.5 w-3.5 text-amber-500 shrink-0 mt-0.5" />
                                <span className="text-amber-800">
                                  Estás sobreescribiendo la auto-detección ({isAutoClassification ? 'Clasificación' : 'Regresión'} → {csvTaskTypeOverride === 'classification' ? 'Clasificación' : 'Regresión'}).
                                  Esto puede ser útil para casos atípicos, por ejemplo: predecir edades (valores enteros pero continuos) como regresión, o predecir ratings (valores numéricos pero discretos) como clasificación.
                                </span>
                              </>
                            ) : (
                              <>
                                <CheckCircle2 className="h-3.5 w-3.5 text-emerald-500 shrink-0 mt-0.5" />
                                <span className="text-emerald-800">
                                  Tu selección coincide con la auto-detección.
                                </span>
                              </>
                            )}
                          </div>
                        )}
                      </div>
                    );
                  })()}

                  {/* Column analysis */}
                  <div className="space-y-2">
                    <h4 className="text-sm font-semibold flex items-center gap-1.5">
                      <Table className="h-4 w-4 text-emerald-500" />
                      Análisis de columnas
                    </h4>
                    <ScrollArea className="max-h-64">
                      <div className="space-y-1">
                        {csvColumnTypes.map((ct, idx) => {
                          const typeColors: Record<string, string> = {
                            numeric: 'bg-blue-100 text-blue-700 border-blue-200',
                            categorical: 'bg-purple-100 text-purple-700 border-purple-200',
                            id: 'bg-gray-100 text-gray-700 border-gray-200',
                            date: 'bg-amber-100 text-amber-700 border-amber-200',
                            empty: 'bg-red-100 text-red-700 border-red-200',
                          };
                          const typeLabels: Record<string, string> = {
                            numeric: 'Numérica',
                            categorical: 'Categórica',
                            id: 'ID',
                            date: 'Fecha',
                            empty: 'Vacía',
                          };
                          const isTarget = csvColumns[idx] === csvTargetColumn;
                          return (
                            <div
                              key={ct.name}
                              className={`flex items-center justify-between p-2 rounded-lg text-xs transition-colors ${
                                isTarget ? 'bg-emerald-50 border border-emerald-200' : 'hover:bg-muted/50'
                              }`}
                            >
                              <div className="flex items-center gap-2 min-w-0 flex-1">
                                <span className="font-medium truncate text-foreground">{ct.name}</span>
                                <Badge
                                  variant="outline"
                                  className={`text-[9px] px-1.5 py-0 shrink-0 ${typeColors[ct.type] || ''}`}
                                >
                                  {typeLabels[ct.type] || ct.type}
                                </Badge>
                                {isTarget && (
                                  <Badge variant="outline" className="text-[9px] px-1.5 py-0 bg-emerald-100 text-emerald-700 border-emerald-200 shrink-0">
                                    Target
                                  </Badge>
                                )}
                              </div>
                              <div className="flex items-center gap-3 text-muted-foreground shrink-0 ml-2">
                                <span>{ct.uniqueValues} únicos</span>
                                {ct.nullCount > 0 && (
                                  <span className="text-amber-600">{ct.nullCount} nulos</span>
                                )}
                              </div>
                            </div>
                          );
                        })}
                      </div>
                    </ScrollArea>
                  </div>

                  {/* Preview table */}
                  {csvPreview.length > 0 && (
                    <div className="space-y-2">
                      <h4 className="text-sm font-semibold flex items-center gap-1.5">
                        <Eye className="h-4 w-4 text-emerald-500" />
                        Vista previa (primeras 5 filas)
                      </h4>
                      <div className="overflow-x-auto border rounded-lg">
                        <table className="w-full text-[10px]">
                          <thead>
                            <tr className="bg-muted/50">
                              {csvColumns.map((col) => (
                                <th
                                  key={col}
                                  className={`py-1.5 px-2 text-left font-semibold whitespace-nowrap ${
                                    col === csvTargetColumn ? 'text-emerald-700 bg-emerald-50' : 'text-foreground'
                                  }`}
                                >
                                  {col}
                                </th>
                              ))}
                            </tr>
                          </thead>
                          <tbody>
                            {csvPreview.slice(0, 5).map((row, ri) => (
                              <tr key={ri} className="border-t">
                                {row.map((val, ci) => (
                                  <td
                                    key={ci}
                                    className={`py-1 px-2 whitespace-nowrap max-w-[120px] truncate ${
                                      csvColumns[ci] === csvTargetColumn ? 'bg-emerald-50/50 font-medium' : ''
                                    }`}
                                  >
                                    {val}
                                  </td>
                                ))}
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    </div>
                  )}
                </div>
              )}
            </div>

            {/* Modal Footer */}
            <div className="flex items-center justify-between p-4 border-t bg-slate-50 shrink-0">
              <Button
                variant="ghost"
                onClick={() => {
                  setShowCSVModal(false);
                  setCsvImportStep('upload');
                  setCsvParseProgress(null);
                }}
              >
                Cancelar
              </Button>
              <div className="flex items-center gap-2">
                {csvImportStep === 'configure' && (
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => {
                      setCsvImportStep('upload');
                      setCsvParseProgress(null);
                    }}
                  >
                    ← Subir otro
                  </Button>
                )}
                {csvImportStep === 'configure' && csvTargetColumn && (
                  <Button
                    onClick={handleCSVConfirm}
                    className="bg-emerald-500 hover:bg-emerald-600 text-white gap-1.5"
                    size="sm"
                  >
                    <CheckCircle2 className="h-4 w-4" />
                    Importar Dataset
                  </Button>
                )}
              </div>
            </div>
          </div>
        </div>
      )}
    </TooltipProvider>
  );
}
