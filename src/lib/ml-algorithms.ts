/**
 * ML HyperLab — Algoritmos de Machine Learning implementados en TypeScript puro
 * Todo corre en el navegador. Sin backend, sin Python, sin instalación.
 */

// ============================================================
// Types
// ============================================================
type Vector = number[];
type Matrix = number[][];
type Label = number;

export interface TrainResult {
  success: boolean;
  metrics: { accuracy: number; precision: number; recall: number; f1: number };
  confusion_matrix: number[][];
  feature_importance: { feature: string; importance: number; index: number }[];
  tree_text: string | null;
  n_train: number;
  n_test: number;
  target_names: string[];
  feature_names: string[];
  error?: string;
}

export interface RegressionTrainResult {
  success: boolean;
  metrics: { mse: number; rmse: number; mae: number; r2: number };
  feature_importance: { feature: string; importance: number; index: number }[];
  tree_text: string | null;
  n_train: number;
  n_test: number;
  target_names: string[];
  feature_names: string[];
  y_test: number[];
  y_pred: number[];
  error?: string;
}

export interface RegressionPlotData {
  points_x: number[];
  points_y: number[];
  points_true: number[];
  points_pred: number[];
  feature_names: string[];
  regression_line_x: number[];
  regression_line_y: number[];
}

export interface BoundaryResult {
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

// ============================================================
// Datasets
// ============================================================
function seedRandom(seed: number) {
  // Simple LCG random number generator
  let s = seed;
  return () => {
    s = (s * 1664525 + 1013904223) & 0xffffffff;
    return (s >>> 0) / 0xffffffff;
  };
}

export function getDataset(name: string): {
  X: Matrix;
  y: Label[];
  feature_names: string[];
  target_names: string[];
} {
  if (name === 'moons') return makeMoons(500, 0.15, 42);
  if (name === 'circles') return makeCircles(500, 0.1, 0.5, 42);
  if (name === 'classification') return makeClassification(500, 42);
  if (name === 'iris') return getIris();
  if (name === 'wine') return getWine();
  if (name === 'breast_cancer') return getBreastCancer();
  if (name === 'make_regression') return makeRegressionDataset(500, 2, 10, 42);
  if (name === 'boston') return getBoston();
  throw new Error(`Dataset desconocido: ${name}`);
}

function makeMoons(n: number, noise: number, seed: number) {
  const rand = seedRandom(seed);
  const randn = () => {
    // Box-Muller transform
    const u1 = rand();
    const u2 = rand();
    return Math.sqrt(-2 * Math.log(u1 + 1e-10)) * Math.cos(2 * Math.PI * u2);
  };
  const X: Matrix = [];
  const y: Label[] = [];
  for (let i = 0; i < n; i++) {
    const label = i < n / 2 ? 0 : 1;
    const angle = (i < n / 2 ? i : i - n / 2) / (n / 2) * Math.PI;
    const x1 = (i < n / 2 ? Math.cos(angle) : 1 - Math.cos(angle)) + randn() * noise;
    const x2 = (i < n / 2 ? Math.sin(angle) : 1 - Math.sin(angle) - 0.5) + randn() * noise;
    X.push([x1, x2]);
    y.push(label);
  }
  return { X, y, feature_names: ['Feature X1', 'Feature X2'], target_names: ['Clase 0', 'Clase 1'] };
}

function makeCircles(n: number, noise: number, factor: number, seed: number) {
  const rand = seedRandom(seed);
  const randn = () => {
    const u1 = rand();
    const u2 = rand();
    return Math.sqrt(-2 * Math.log(u1 + 1e-10)) * Math.cos(2 * Math.PI * u2);
  };
  const X: Matrix = [];
  const y: Label[] = [];
  for (let i = 0; i < n; i++) {
    const label = i < n / 2 ? 0 : 1;
    const angle = rand() * 2 * Math.PI;
    const r = label === 0 ? 1 : factor;
    const x1 = r * Math.cos(angle) + randn() * noise;
    const x2 = r * Math.sin(angle) + randn() * noise;
    X.push([x1, x2]);
    y.push(label);
  }
  return { X, y, feature_names: ['Feature X1', 'Feature X2'], target_names: ['Clase 0', 'Clase 1'] };
}

function makeClassification(n: number, seed: number) {
  const rand = seedRandom(seed);
  const randn = () => {
    const u1 = rand();
    const u2 = rand();
    return Math.sqrt(-2 * Math.log(u1 + 1e-10)) * Math.cos(2 * Math.PI * u2);
  };
  const X: Matrix = [];
  const y: Label[] = [];
  // Two clusters
  for (let i = 0; i < n; i++) {
    const label = i < n / 2 ? 0 : 1;
    const cx = label === 0 ? -1 : 1;
    const cy = label === 0 ? -1 : 1;
    X.push([cx + randn() * 0.8, cy + randn() * 0.8]);
    y.push(label);
  }
  return { X, y, feature_names: ['Feature X1', 'Feature X2'], target_names: ['Clase 0', 'Clase 1'] };
}

// Real datasets embedded as compressed arrays
function getIris() {
  // Iris dataset - 150 samples, 4 features, 3 classes
  const raw = `5.1,3.5,1.4,0.2,0
4.9,3,1.4,0.2,0
4.7,3.2,1.3,0.2,0
4.6,3.1,1.5,0.2,0
5,3.6,1.4,0.2,0
5.4,3.9,1.7,0.4,0
4.6,3.4,1.4,0.3,0
5,3.4,1.5,0.2,0
4.4,2.9,1.4,0.2,0
4.9,3.1,1.5,0.1,0
5.4,3.7,1.5,0.2,0
4.8,3.4,1.6,0.2,0
4.8,3,1.4,0.1,0
4.3,3,1.1,0.1,0
5.8,4,1.2,0.2,0
5.7,4.4,1.5,0.4,0
5.4,3.9,1.3,0.4,0
5.1,3.5,1.4,0.3,0
5.7,3.8,1.7,0.3,0
5.1,3.8,1.5,0.3,0
5.4,3.4,1.7,0.2,0
5.1,3.7,1.5,0.4,0
4.6,3.6,1,0.2,0
5.1,3.3,1.7,0.5,0
4.8,3.4,1.9,0.2,0
5,3,1.6,0.2,0
5,3.4,1.6,0.4,0
5.2,3.5,1.5,0.2,0
5.2,3.4,1.4,0.2,0
4.7,3.2,1.6,0.2,0
4.8,3.1,1.6,0.2,0
5.4,3.4,1.5,0.4,0
5.2,4.1,1.5,0.1,0
5.5,4.2,1.4,0.2,0
4.9,3.1,1.5,0.2,0
5,3.2,1.2,0.2,0
5.5,3.5,1.3,0.2,0
4.9,3.6,1.4,0.1,0
4.4,3,1.3,0.2,0
5.1,3.4,1.5,0.2,0
5,3.5,1.3,0.3,0
4.5,2.3,1.3,0.3,0
4.4,3.2,1.3,0.2,0
5,3.5,1.6,0.6,0
5.1,3.8,1.9,0.4,0
4.8,3,1.4,0.3,0
5.1,3.8,1.6,0.2,0
4.6,3.2,1.4,0.2,0
5.3,3.7,1.5,0.2,0
5,3.3,1.4,0.2,0
7,3.2,4.7,1.4,1
6.4,3.2,4.5,1.5,1
6.9,3.1,4.9,1.5,1
5.5,2.3,4,1.3,1
6.5,2.8,4.6,1.5,1
5.7,2.8,4.5,1.3,1
6.3,3.3,4.7,1.6,1
4.9,2.4,3.3,1,1
6.6,2.9,4.6,1.3,1
5.2,2.7,3.9,1.4,1
5,2,3.5,1,1
5.9,3,4.2,1.5,1
6,2.2,4,1,1
6.1,2.9,4.7,1.4,1
5.6,2.9,3.6,1.3,1
6.7,3.1,4.4,1.4,1
5.6,3,4.5,1.5,1
5.8,2.7,4.1,1,1
6.2,2.2,4.5,1.5,1
5.6,2.5,3.9,1.1,1
5.9,3.2,4.8,1.8,1
6.1,2.8,4,1.3,1
6.3,2.5,4.9,1.5,1
6.1,2.8,4.7,1.2,1
6.4,2.9,4.3,1.3,1
6.6,3,4.4,1.4,1
6.8,2.8,4.8,1.4,1
6.7,3,5,1.7,1
6,2.9,4.5,1.5,1
5.7,2.6,3.5,1,1
5.5,2.4,3.8,1.1,1
5.5,2.4,3.7,1,1
5.8,2.7,3.9,1.2,1
6,2.7,5.1,1.6,1
5.4,3,4.5,1.5,1
6,3.4,4.5,1.6,1
6.7,3.1,4.7,1.5,1
6.3,2.3,4.4,1.3,1
5.6,3,4.1,1.3,1
5.5,2.5,4,1.3,1
5.5,2.6,4.4,1.2,1
6.1,3,4.6,1.4,1
5.8,2.6,4,1.2,1
5,2.3,3.3,1,1
5.6,2.7,4.2,1.3,1
5.7,3,4.2,1.2,1
5.7,2.9,4.2,1.3,1
6.2,2.9,4.3,1.3,1
5.1,2.5,3,1.1,1
5.7,2.8,4.1,1.3,1
6.3,3.3,6,2.5,2
5.8,2.7,5.1,1.9,2
7.1,3,5.9,2.1,2
6.3,2.9,5.6,1.8,2
6.5,3,5.8,2.2,2
7.6,3,6.6,2.1,2
4.9,2.5,4.5,1.7,2
7.3,2.9,6.3,1.8,2
6.7,2.5,5.8,1.8,2
7.2,3.6,6.1,2.5,2
6.5,3.2,5.1,2,2
6.4,2.7,5.3,1.9,2
6.8,3,5.5,2.1,2
5.7,2.5,5,2,2
5.8,2.8,5.1,2.4,2
6.4,3.2,5.3,2.3,2
6.5,3,5.5,1.8,2
7.7,3.8,6.7,2.2,2
7.7,2.6,6.9,2.3,2
6,2.2,5,1.5,2
6.9,3.2,5.7,2.3,2
5.6,2.8,4.9,2,2
7.7,2.8,6.7,2,2
6.3,2.7,4.9,1.8,2
6.7,3.3,5.7,2.1,2
7.2,3.2,6,1.8,2
6.2,2.8,4.8,1.8,2
6.1,3,4.9,1.8,2
6.4,2.8,5.6,2.1,2
7.2,3,5.8,1.6,2
7.4,2.8,6.1,1.9,2
7.9,3.8,6.4,2,2
6.4,2.8,5.6,2.2,2
6.3,2.8,5.1,1.5,2
6.1,2.6,5.6,1.4,2
7.7,3,6.1,2.3,2
6.3,3.4,5.6,2.4,2
6.4,3.1,5.5,1.8,2
6,3,4.8,1.8,2
6.9,3.1,5.4,2.1,2
6.7,3.1,5.6,2.4,2
6.9,3.1,5.1,2.3,2
5.8,2.7,5.1,1.9,2
6.8,3.2,5.9,2.3,2
6.7,3.3,5.7,2.5,2
6.7,3,5.2,2.3,2
6.3,2.5,5,1.9,2
6.5,3,5.2,2,2
6.2,3.4,5.4,2.3,2
5.9,3,5.1,1.8,2`;
  const lines = raw.trim().split('\n');
  const X: Matrix = [];
  const y: Label[] = [];
  for (const line of lines) {
    const parts = line.split(',').map(Number);
    y.push(parts[4]);
    X.push(parts.slice(0, 4));
  }
  return { X, y, feature_names: ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)'], target_names: ['setosa', 'versicolor', 'virginica'] };
}

function getWine() {
  // Simplified wine dataset - 13 features, 3 classes, 178 samples
  // Using a representative subset for browser performance
  const rand = seedRandom(42);
  const nFeatures = 13;
  const featureNames = ['alcohol', 'malic acid', 'ash', 'alcalinity', 'magnesium', 'total phenols', 'flavanoids', 'nonflavanoid phenols', 'proanthocyanins', 'color intensity', 'hue', 'OD280/OD315', 'proline'];
  // Generate synthetic wine-like data
  const centers = [
    [13.7, 2.0, 2.4, 17.0, 106, 2.8, 3.0, 0.3, 2.2, 5.6, 1.05, 3.4, 1050],
    [12.3, 1.9, 2.4, 21.0, 94, 2.1, 1.7, 0.4, 1.4, 4.0, 0.95, 2.7, 600],
    [13.2, 3.3, 2.3, 22.5, 98, 1.7, 1.0, 0.5, 0.9, 7.0, 0.6, 1.7, 500],
  ];
  const X: Matrix = [];
  const y: Label[] = [];
  for (let c = 0; c < 3; c++) {
    const count = c === 0 ? 59 : c === 1 ? 71 : 48;
    for (let i = 0; i < count; i++) {
      const row = centers[c].map((v, fi) => {
        const spread = fi === 0 ? 0.5 : fi === 12 ? 100 : fi === 4 ? 10 : v * 0.15;
        return v + (rand() - 0.5) * 2 * spread;
      });
      X.push(row);
      y.push(c);
    }
  }
  return { X, y, feature_names: featureNames, target_names: ['class_0', 'class_1', 'class_2'] };
}

function getBreastCancer() {
  // Simplified breast cancer dataset - 30 features, 2 classes, 569 samples
  const rand = seedRandom(42);
  const featureNames = [
    'mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean smoothness',
    'mean compactness', 'mean concavity', 'mean concave points', 'mean symmetry', 'mean fractal dim',
    'radius error', 'texture error', 'perimeter error', 'area error', 'smoothness error',
    'compactness error', 'concavity error', 'concave points error', 'symmetry error', 'fractal dim error',
    'worst radius', 'worst texture', 'worst perimeter', 'worst area', 'worst smoothness',
    'worst compactness', 'worst concavity', 'worst concave points', 'worst symmetry', 'worst fractal dim',
  ];
  const malignantCenter = featureNames.map((_, i) => i === 0 ? 17 : i < 10 ? 5 + rand() * 10 : i < 20 ? 1 + rand() * 3 : 20 + rand() * 10);
  const benignCenter = featureNames.map((_, i) => i === 0 ? 12 : i < 10 ? 2 + rand() * 5 : i < 20 ? 0.3 + rand() * 1 : 13 + rand() * 5);
  const X: Matrix = [];
  const y: Label[] = [];
  for (let i = 0; i < 569; i++) {
    const label = i < 212 ? 0 : 1;
    const center = label === 0 ? malignantCenter : benignCenter;
    const row = center.map((v) => v + (rand() - 0.5) * v * 0.4);
    X.push(row);
    y.push(label);
  }
  return { X, y, feature_names: featureNames, target_names: ['malignant', 'benign'] };
}

// ============================================================
// Utility Functions
// ============================================================
function shuffleArray<T>(arr: T[], seed: number): T[] {
  const rand = seedRandom(seed);
  const result = [...arr];
  for (let i = result.length - 1; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1));
    [result[i], result[j]] = [result[j], result[i]];
  }
  return result;
}

function trainTestSplit(X: Matrix, y: Label[], testSize: number, seed: number) {
  const n = X.length;
  const indices = shuffleArray(Array.from({ length: n }, (_, i) => i), seed);
  const splitIdx = Math.floor(n * (1 - testSize));
  const trainIdx = indices.slice(0, splitIdx);
  const testIdx = indices.slice(splitIdx);
  // Stratified split
  const classes = [...new Set(y)];
  const trainIdx2: number[] = [];
  const testIdx2: number[] = [];
  for (const c of classes) {
    const classIndices = indices.filter((i) => y[i] === c);
    const classSplit = Math.floor(classIndices.length * (1 - testSize));
    trainIdx2.push(...classIndices.slice(0, classSplit));
    testIdx2.push(...classIndices.slice(classSplit));
  }
  return {
    X_train: trainIdx2.map((i) => X[i]),
    X_test: testIdx2.map((i) => X[i]),
    y_train: trainIdx2.map((i) => y[i]),
    y_test: testIdx2.map((i) => y[i]),
  };
}

function standardScale(X: Matrix): { scaled: Matrix; means: Vector; stds: Vector } {
  const n = X.length;
  const p = X[0].length;
  const means: Vector = new Array(p).fill(0);
  const stds: Vector = new Array(p).fill(0);
  for (let j = 0; j < p; j++) {
    for (let i = 0; i < n; i++) means[j] += X[i][j];
    means[j] /= n;
    for (let i = 0; i < n; i++) stds[j] += (X[i][j] - means[j]) ** 2;
    stds[j] = Math.sqrt(stds[j] / n) || 1;
  }
  const scaled = X.map((row) => row.map((v, j) => (v - means[j]) / stds[j]));
  return { scaled, means, stds };
}

function scaleWith(X: Matrix, means: Vector, stds: Vector): Matrix {
  return X.map((row) => row.map((v, j) => (v - means[j]) / stds[j]));
}

function euclidean(a: Vector, b: Vector): number {
  let sum = 0;
  for (let i = 0; i < a.length; i++) sum += (a[i] - b[i]) ** 2;
  return Math.sqrt(sum);
}

function manhattan(a: Vector, b: Vector): number {
  let sum = 0;
  for (let i = 0; i < a.length; i++) sum += Math.abs(a[i] - b[i]);
  return sum;
}

function mode(arr: number[]): number {
  const counts = new Map<number, number>();
  for (const v of arr) counts.set(v, (counts.get(v) || 0) + 1);
  let maxCount = 0;
  let mode = arr[0];
  for (const [val, count] of counts) {
    if (count > maxCount) { maxCount = count; mode = val; }
  }
  return mode;
}

function sigmoid(x: number): number {
  if (x > 500) return 1;
  if (x < -500) return 0;
  return 1 / (1 + Math.exp(-x));
}

function relu(x: number): number {
  return Math.max(0, x);
}

function tanh(x: number): number {
  return Math.tanh(x);
}

function softmax(arr: number[]): number[] {
  const max = Math.max(...arr);
  const exps = arr.map((v) => Math.exp(v - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((v) => v / sum);
}

function computeMetrics(yTrue: Label[], yPred: Label[], nClasses: number) {
  const n = yTrue.length;
  let correct = 0;
  for (let i = 0; i < n; i++) if (yTrue[i] === yPred[i]) correct++;
  const accuracy = correct / n;

  // Per-class precision, recall, then weighted average
  let precision = 0, recall = 0, f1 = 0;
  let totalWeight = 0;
  for (let c = 0; c < nClasses; c++) {
    const tp = yPred.filter((p, i) => p === c && yTrue[i] === c).length;
    const fp = yPred.filter((p, i) => p === c && yTrue[i] !== c).length;
    const fn = yPred.filter((_, i) => yTrue[i] === c && yPred[i] !== c).length;
    const classCount = yTrue.filter((y) => y === c).length;
    const p_c = tp + fp > 0 ? tp / (tp + fp) : 0;
    const r_c = tp + fn > 0 ? tp / (tp + fn) : 0;
    const f1_c = p_c + r_c > 0 ? 2 * p_c * r_c / (p_c + r_c) : 0;
    precision += p_c * classCount;
    recall += r_c * classCount;
    f1 += f1_c * classCount;
    totalWeight += classCount;
  }
  return {
    accuracy: round4(accuracy),
    precision: round4(precision / totalWeight),
    recall: round4(recall / totalWeight),
    f1: round4(f1 / totalWeight),
  };
}

function computeConfusionMatrix(yTrue: Label[], yPred: Label[], nClasses: number): number[][] {
  const cm = Array.from({ length: nClasses }, () => new Array(nClasses).fill(0));
  for (let i = 0; i < yTrue.length; i++) cm[yTrue[i]][yPred[i]]++;
  return cm;
}

function round4(n: number): number {
  return Math.round(n * 10000) / 10000;
}

// ============================================================
// Decision Tree (used by RF and XGBoost too)
// ============================================================
interface TreeNode {
  feature?: number;
  threshold?: number;
  left?: TreeNode;
  right?: TreeNode;
  label?: number;
  samples?: number;
  impurity?: number;
  featureImportance?: number[];
}

function giniImpurity(y: Label[]): number {
  const n = y.length;
  if (n === 0) return 0;
  const counts = new Map<number, number>();
  for (const v of y) counts.set(v, (counts.get(v) || 0) + 1);
  let impurity = 1;
  for (const count of counts.values()) {
    const p = count / n;
    impurity -= p * p;
  }
  return impurity;
}

function entropyImpurity(y: Label[]): number {
  const n = y.length;
  if (n === 0) return 0;
  const counts = new Map<number, number>();
  for (const v of y) counts.set(v, (counts.get(v) || 0) + 1);
  let ent = 0;
  for (const count of counts.values()) {
    const p = count / n;
    if (p > 0) ent -= p * Math.log2(p);
  }
  return ent;
}

function buildDecisionTree(
  X: Matrix, y: Label[], maxDepth: number | null, minSamplesSplit: number,
  minSamplesLeaf: number, criterion: 'gini' | 'entropy', nFeatures: number | null,
  depth: number = 0, rand?: () => number
): TreeNode {
  const nSamples = y.length;
  const impurityFn = criterion === 'gini' ? giniImpurity : entropyImpurity;

  // Leaf conditions
  if (nSamples <= minSamplesSplit || (maxDepth !== null && depth >= maxDepth) || new Set(y).size === 1) {
    return { label: mode(y), samples: nSamples };
  }

  const nFeaturesTotal = X[0].length;
  const featureIndices = nFeatures && nFeatures < nFeaturesTotal
    ? shuffleArray(Array.from({ length: nFeaturesTotal }, (_, i) => i), depth * 1000 + nSamples).slice(0, nFeatures)
    : Array.from({ length: nFeaturesTotal }, (_, i) => i);

  let bestFeature = -1;
  let bestThreshold = 0;
  let bestImpurity = Infinity;
  let bestLeftIdx: number[] = [];
  let bestRightIdx: number[] = [];

  for (const fi of featureIndices) {
    const values = [...new Set(X.map((row) => row[fi]))].sort((a, b) => a - b);
    for (let t = 0; t < values.length - 1; t++) {
      const threshold = (values[t] + values[t + 1]) / 2;
      const leftIdx: number[] = [];
      const rightIdx: number[] = [];
      for (let i = 0; i < nSamples; i++) {
        if (X[i][fi] <= threshold) leftIdx.push(i);
        else rightIdx.push(i);
      }
      if (leftIdx.length < minSamplesLeaf || rightIdx.length < minSamplesLeaf) continue;
      const leftY = leftIdx.map((i) => y[i]);
      const rightY = rightIdx.map((i) => y[i]);
      const weightedImpurity = (leftY.length * impurityFn(leftY) + rightY.length * impurityFn(rightY)) / nSamples;
      if (weightedImpurity < bestImpurity) {
        bestImpurity = weightedImpurity;
        bestFeature = fi;
        bestThreshold = threshold;
        bestLeftIdx = leftIdx;
        bestRightIdx = rightIdx;
      }
    }
  }

  if (bestFeature === -1) return { label: mode(y), samples: nSamples };

  const left = buildDecisionTree(
    bestLeftIdx.map((i) => X[i]), bestLeftIdx.map((i) => y[i]),
    maxDepth, minSamplesSplit, minSamplesLeaf, criterion, nFeatures, depth + 1, rand
  );
  const right = buildDecisionTree(
    bestRightIdx.map((i) => X[i]), bestRightIdx.map((i) => y[i]),
    maxDepth, minSamplesSplit, minSamplesLeaf, criterion, nFeatures, depth + 1, rand
  );

  return { feature: bestFeature, threshold: bestThreshold, left, right, samples: nSamples, impurity: impurityFn(y) };
}

function predictTree(node: TreeNode, x: Vector): number {
  if (node.label !== undefined) return node.label;
  if (x[node.feature!] <= node.threshold!) return predictTree(node.left!, x);
  return predictTree(node.right!, x);
}

function treeFeatureImportance(node: TreeNode, nFeatures: number): number[] {
  const importance = new Array(nFeatures).fill(0);
  function traverse(n: TreeNode) {
    if (n.feature !== undefined) {
      const leftSize = n.left?.samples || 0;
      const rightSize = n.right?.samples || 0;
      importance[n.feature] += (leftSize + rightSize) * (n.impurity || 0);
      traverse(n.left!);
      traverse(n.right!);
    }
  }
  traverse(node);
  const total = importance.reduce((a, b) => a + b, 0) || 1;
  return importance.map((v) => v / total);
}

function treeToText(node: TreeNode, featureNames: string[], depth: number = 0, prefix: string = ''): string {
  if (node.label !== undefined) return `${prefix}|--- class: ${node.label}\n`;
  const fname = featureNames[node.feature!] || `Feature ${node.feature}`;
  let text = `${prefix}|--- ${fname} <= ${node.threshold!.toFixed(2)}\n`;
  text += treeToText(node.left!, featureNames, depth + 1, prefix + '|   ');
  text += `${prefix}|--- ${fname} >  ${node.threshold!.toFixed(2)}\n`;
  text += treeToText(node.right!, featureNames, depth + 1, prefix + '|   ');
  return text;
}

// ============================================================
// KNN
// ============================================================
class KNNClassifier {
  private X: Matrix = [];
  private y: Label[] = [];

  constructor(private k: number = 5, private weights: string = 'uniform', private p: number = 2) {}

  fit(X: Matrix, y: Label[]) { this.X = X; this.y = y; }

  predict(X: Matrix): Label[] {
    return X.map((x) => this.predictOne(x));
  }

  private predictOne(x: Vector): number {
    const distFn = this.p === 1 ? manhattan : euclidean;
    const distances = this.X.map((trainX, i) => ({ dist: distFn(x, trainX), label: this.y[i] }));
    distances.sort((a, b) => a.dist - b.dist);
    const neighbors = distances.slice(0, this.k);
    if (this.weights === 'distance') {
      const weightedCounts = new Map<number, number>();
      for (const n of neighbors) {
        const w = 1 / (n.dist + 1e-10);
        weightedCounts.set(n.label, (weightedCounts.get(n.label) || 0) + w);
      }
      let best = neighbors[0].label;
      let bestW = 0;
      for (const [label, w] of weightedCounts) {
        if (w > bestW) { bestW = w; best = label; }
      }
      return best;
    }
    return mode(neighbors.map((n) => n.label));
  }
}

// ============================================================
// Logistic Regression
// ============================================================
class LogisticRegressionClassifier {
  private weights: Matrix = [];
  private biases: Vector = [];
  private nClasses: number = 2;

  constructor(private C: number = 1.0, private maxIter: number = 100) {}

  fit(X: Matrix, y: Label[]) {
    const n = X.length;
    const p = X[0].length;
    this.nClasses = new Set(y).size;
    const classes = [...new Set(y)].sort((a, b) => a - b);
    const lr = 0.01;
    const reg = 1 / this.C;

    this.weights = Array.from({ length: this.nClasses }, () => new Array(p).fill(0));
    this.biases = new Array(this.nClasses).fill(0);

    for (let iter = 0; iter < this.maxIter; iter++) {
      for (let i = 0; i < n; i++) {
        const logits = classes.map((c, ci) => {
          let s = this.biases[ci];
          for (let j = 0; j < p; j++) s += this.weights[ci][j] * X[i][j];
          return s;
        });
        const probs = this.nClasses === 2
          ? [1 - sigmoid(logits[0]), sigmoid(logits[0])]
          : softmax(logits);

        const trueClass = classes.indexOf(y[i]);
        for (let ci = 0; ci < this.nClasses; ci++) {
          const error = (ci === trueClass ? 1 : 0) - probs[ci];
          for (let j = 0; j < p; j++) {
            this.weights[ci][j] += lr * (error * X[i][j] - reg * this.weights[ci][j]);
          }
          this.biases[ci] += lr * error;
        }
      }
    }
  }

  predict(X: Matrix): Label[] {
    const classes = [...Array(this.nClasses).keys()];
    return X.map((x) => {
      const logits = classes.map((c) => {
        let s = this.biases[c];
        for (let j = 0; j < x.length; j++) s += this.weights[c][j] * x[j];
        return s;
      });
      let bestClass = 0;
      let bestLogit = -Infinity;
      for (let c = 0; c < this.nClasses; c++) {
        if (logits[c] > bestLogit) { bestLogit = logits[c]; bestClass = c; }
      }
      return bestClass;
    });
  }

  getFeatureImportance(): number[] {
    return this.weights.map((w) => w.map(Math.abs)).reduce((acc, w) => w.map((v, i) => acc[i] + v), new Array(this.weights[0].length).fill(0));
  }
}

// ============================================================
// Neural Network (MLP)
// ============================================================
class MLPClassifier {
  private layerWeights: Matrix[] = [];
  private layerBiases: Vector[] = [];
  private nClasses: number = 2;
  private hiddenSizes: number[];

  constructor(
    hiddenLayerSizes: number[] = [100],
    private activation: string = 'relu',
    private alpha: number = 0.0001,
    private learningRate: number = 0.001,
    private maxIter: number = 200
  ) {
    this.hiddenSizes = hiddenLayerSizes;
  }

  fit(X: Matrix, y: Label[]) {
    const n = X.length;
    const p = X[0].length;
    this.nClasses = new Set(y).size;
    const layerSizes = [p, ...this.hiddenSizes, this.nClasses];
    const rand = seedRandom(42);

    // Initialize weights
    this.layerWeights = [];
    this.layerBiases = [];
    for (let l = 0; l < layerSizes.length - 1; l++) {
      const fan = layerSizes[l];
      const scale = Math.sqrt(2 / fan);
      this.layerWeights.push(Array.from({ length: layerSizes[l + 1] }, () =>
        Array.from({ length: layerSizes[l] }, () => (rand() - 0.5) * 2 * scale)
      ));
      this.layerBiases.push(new Array(layerSizes[l + 1]).fill(0));
    }

    const activationFn = this.activation === 'tanh' ? tanh : this.activation === 'logistic' ? sigmoid : relu;
    const activationDeriv = this.activation === 'tanh'
      ? (x: number) => 1 - Math.tanh(x) ** 2
      : this.activation === 'logistic'
        ? (x: number) => { const s = sigmoid(x); return s * (1 - s); }
        : (x: number) => x > 0 ? 1 : 0;

    // Training loop (mini-batch SGD)
    for (let iter = 0; iter < this.maxIter; iter++) {
      for (let i = 0; i < n; i++) {
        // Forward pass
        const activations: Matrix = [X[i]];
        const preActivations: Matrix = [];
        for (let l = 0; l < this.layerWeights.length; l++) {
          const input = activations[l];
          const pre: Vector = [];
          for (let j = 0; j < this.layerWeights[l].length; j++) {
            let sum = this.layerBiases[l][j];
            for (let k = 0; k < input.length; k++) sum += this.layerWeights[l][j][k] * input[k];
            pre.push(sum);
          }
          preActivations.push(pre);
          if (l < this.layerWeights.length - 1) {
            activations.push(pre.map(activationFn));
          } else {
            activations.push(this.nClasses === 2 ? [sigmoid(pre[0])] : softmax(pre));
          }
        }

        // Compute output error
        const output = activations[activations.length - 1];
        const trueClass = y[i];
        const errors: Vector = output.map((o, j) => {
          const target = j === trueClass ? 1 : 0;
          return target - o;
        });

        // Backward pass
        const deltas: Vector[] = [errors];
        for (let l = this.layerWeights.length - 2; l >= 0; l--) {
          const delta: Vector = [];
          for (let j = 0; j < this.layerWeights[l].length; j++) {
            let sum = 0;
            for (let k = 0; k < this.layerWeights[l + 1].length; k++) {
              sum += this.layerWeights[l + 1][k][j] * deltas[0][k];
            }
            delta.push(sum * activationDeriv(preActivations[l][j]));
          }
          deltas.unshift(delta);
        }

        // Update weights
        for (let l = 0; l < this.layerWeights.length; l++) {
          for (let j = 0; j < this.layerWeights[l].length; j++) {
            for (let k = 0; k < this.layerWeights[l][j].length; k++) {
              this.layerWeights[l][j][k] += this.learningRate * (deltas[l][j] * activations[l][k] - this.alpha * this.layerWeights[l][j][k]);
            }
            this.layerBiases[l][j] += this.learningRate * deltas[l][j];
          }
        }
      }
    }
  }

  predict(X: Matrix): Label[] {
    const activationFn = this.activation === 'tanh' ? tanh : this.activation === 'logistic' ? sigmoid : relu;
    return X.map((x) => {
      let current = x;
      for (let l = 0; l < this.layerWeights.length; l++) {
        const next: Vector = [];
        for (let j = 0; j < this.layerWeights[l].length; j++) {
          let sum = this.layerBiases[l][j];
          for (let k = 0; k < current.length; k++) sum += this.layerWeights[l][j][k] * current[k];
          if (l < this.layerWeights.length - 1) next.push(activationFn(sum));
          else next.push(sum);
        }
        current = next;
      }
      let bestClass = 0;
      let bestVal = -Infinity;
      for (let c = 0; c < current.length; c++) {
        if (current[c] > bestVal) { bestVal = current[c]; bestClass = c; }
      }
      return bestClass;
    });
  }
}

// ============================================================
// MAIN: Train a model (with dataset name — built-in datasets)
// ============================================================
export function trainModel(
  algorithm: string,
  datasetName: string,
  hyperparams: Record<string, unknown>,
  testSize: number = 0.3
): TrainResult {
  try {
    const { X, y, feature_names, target_names } = getDataset(datasetName);
    return trainModelWithData(algorithm, X, y, feature_names, target_names, hyperparams, testSize);
  } catch (error: unknown) {
    return {
      success: false,
      metrics: { accuracy: 0, precision: 0, recall: 0, f1: 0 },
      confusion_matrix: [],
      feature_importance: [],
      tree_text: null,
      n_train: 0,
      n_test: 0,
      target_names: [],
      feature_names: [],
      error: error instanceof Error ? error.message : 'Error desconocido',
    };
  }
}

// ============================================================
// Train a model with custom dataset data (CSV import)
// ============================================================
export function trainModelWithData(
  algorithm: string,
  X: Matrix,
  y: Label[],
  feature_names: string[],
  target_names: string[],
  hyperparams: Record<string, unknown>,
  testSize: number = 0.3
): TrainResult {
  try {
    const nClasses = new Set(y).size;
    const { X_train, X_test, y_train, y_test } = trainTestSplit(X, y, testSize, 42);
    const { scaled: X_train_s, means, stds } = standardScale(X_train);
    const X_test_s = scaleWith(X_test, means, stds);

    let y_pred: Label[];
    let featureImportance: { feature: string; importance: number; index: number }[] = [];
    let treeText: string | null = null;

    switch (algorithm) {
      case 'knn': {
        const model = new KNNClassifier(
          Number(hyperparams.n_neighbors || 5),
          String(hyperparams.weights || 'uniform'),
          Number(hyperparams.p || 2)
        );
        model.fit(X_train_s, y_train);
        y_pred = model.predict(X_test_s);
        break;
      }

      case 'decision_tree': {
        const maxDepth = hyperparams.max_depth == null || hyperparams.max_depth === null || String(hyperparams.max_depth) === 'null'
          ? null : Number(hyperparams.max_depth);
        const tree = buildDecisionTree(
          X_train_s, y_train, maxDepth && maxDepth > 0 ? maxDepth : null,
          Number(hyperparams.min_samples_split || 2),
          Number(hyperparams.min_samples_leaf || 1),
          String(hyperparams.criterion || 'gini') as 'gini' | 'entropy',
          null
        );
        y_pred = X_test_s.map((x) => predictTree(tree, x));
        const imp = treeFeatureImportance(tree, feature_names.length);
        featureImportance = feature_names.map((f, i) => ({ feature: f, importance: round4(imp[i]), index: i }))
          .sort((a, b) => b.importance - a.importance);
        treeText = treeToText(tree, feature_names);
        break;
      }

      case 'random_forest': {
        const nEstimators = Number(hyperparams.n_estimators || 100);
        const maxDepth = hyperparams.max_depth == null || String(hyperparams.max_depth) === 'null'
          ? null : Number(hyperparams.max_depth);
        const maxFeatures = String(hyperparams.max_features || 'sqrt') === 'sqrt'
          ? Math.floor(Math.sqrt(feature_names.length))
          : String(hyperparams.max_features) === 'log2'
            ? Math.floor(Math.log2(feature_names.length))
            : Math.floor(Number(hyperparams.max_features || 1) * feature_names.length) || feature_names.length;
        const predictions: number[][] = [];
        const impAvg = new Array(feature_names.length).fill(0);
        for (let t = 0; t < nEstimators; t++) {
          // Bootstrap sample
          const rand = seedRandom(t + 42);
          const bootstrapIdx = Array.from({ length: X_train_s.length }, () => Math.floor(rand() * X_train_s.length));
          const bX = bootstrapIdx.map((i) => X_train_s[i]);
          const by = bootstrapIdx.map((i) => y_train[i]);
          const tree = buildDecisionTree(bX, by, maxDepth && maxDepth > 0 ? maxDepth : null,
            Number(hyperparams.min_samples_split || 2), Number(hyperparams.min_samples_leaf || 1),
            'gini', maxFeatures, 0, rand
          );
          predictions.push(X_test_s.map((x) => predictTree(tree, x)));
          const imp = treeFeatureImportance(tree, feature_names.length);
          for (let j = 0; j < imp.length; j++) impAvg[j] += imp[j];
        }
        y_pred = X_test_s.map((_, i) => mode(predictions.map((p) => p[i])));
        for (let j = 0; j < impAvg.length; j++) impAvg[j] /= nEstimators;
        featureImportance = feature_names.map((f, i) => ({ feature: f, importance: round4(impAvg[i]), index: i }))
          .sort((a, b) => b.importance - a.importance);
        break;
      }

      case 'xgboost': {
        const nEstimators = Number(hyperparams.n_estimators || 100);
        const maxDepth = Number(hyperparams.max_depth || 6);
        const learningRate = Number(hyperparams.learning_rate || 0.3);
        const nTrain = X_train_s.length;
        const nFeatures = X_train_s[0].length;
        // One-hot encode y for multi-class
        const yOneHot = y_train.map((c) => {
          const v = new Array(nClasses).fill(0);
          v[c] = 1;
          return v;
        });
        let residuals = yOneHot.map((r) => [...r]);
        const treePredictions: { tree: TreeNode; classIdx: number }[] = [];
        const impAvg = new Array(nFeatures).fill(0);
        let treesCount = 0;

        for (let round = 0; round < nEstimators; round++) {
          for (let c = 0; c < nClasses; c++) {
            const yResid = residuals.map((r) => r[c] > 0.5 ? 1 : 0);
            const tree = buildDecisionTree(X_train_s, yResid, maxDepth, 2, 1, 'gini', null);
            treePredictions.push({ tree, classIdx: c });
            const imp = treeFeatureImportance(tree, nFeatures);
            for (let j = 0; j < imp.length; j++) impAvg[j] += imp[j];
            treesCount++;
            // Update residuals
            for (let i = 0; i < nTrain; i++) {
              const pred = predictTree(tree, X_train_s[i]);
              residuals[i][c] = yOneHot[i][c] - learningRate * pred;
            }
          }
        }

        // Predict
        y_pred = X_test_s.map((x) => {
          const scores = new Array(nClasses).fill(0);
          for (const { tree, classIdx } of treePredictions) {
            scores[classIdx] += learningRate * predictTree(tree, x);
          }
          return scores.indexOf(Math.max(...scores));
        });

        for (let j = 0; j < impAvg.length; j++) impAvg[j] /= treesCount || 1;
        featureImportance = feature_names.map((f, i) => ({ feature: f, importance: round4(impAvg[i]), index: i }))
          .sort((a, b) => b.importance - a.importance);
        break;
      }

      case 'neural_network': {
        const hiddenSizes = String(hyperparams.hidden_layer_sizes || '100').split(',').map(Number);
        const model = new MLPClassifier(
          hiddenSizes,
          String(hyperparams.activation || 'relu'),
          Number(hyperparams.alpha || 0.0001),
          Number(hyperparams.learning_rate_init || 0.001),
          Number(hyperparams.max_iter || 200)
        );
        model.fit(X_train_s, y_train);
        y_pred = model.predict(X_test_s);
        // No direct feature importance for NN
        break;
      }

      case 'logistic_regression': {
        const model = new LogisticRegressionClassifier(
          Number(hyperparams.C || 1.0),
          Number(hyperparams.max_iter || 100)
        );
        model.fit(X_train_s, y_train);
        y_pred = model.predict(X_test_s);
        const imp = model.getFeatureImportance();
        featureImportance = feature_names.map((f, i) => ({ feature: f, importance: round4(imp[i] / imp.reduce((a, b) => a + b, 0)), index: i }))
          .sort((a, b) => b.importance - a.importance);
        break;
      }

      default:
        throw new Error(`Algoritmo desconocido: ${algorithm}`);
    }

    const metrics = computeMetrics(y_test, y_pred, nClasses);
    const confusion_matrix = computeConfusionMatrix(y_test, y_pred, nClasses);

    return {
      success: true,
      metrics,
      confusion_matrix,
      feature_importance: featureImportance,
      tree_text: treeText,
      n_train: X_train.length,
      n_test: X_test.length,
      target_names,
      feature_names,
    };
  } catch (error: unknown) {
    return {
      success: false,
      metrics: { accuracy: 0, precision: 0, recall: 0, f1: 0 },
      confusion_matrix: [],
      feature_importance: [],
      tree_text: null,
      n_train: 0,
      n_test: 0,
      target_names: [],
      feature_names: [],
      error: error instanceof Error ? error.message : 'Error desconocido',
    };
  }
}

// ============================================================
// Decision Boundary
// ============================================================
export function computeDecisionBoundary(
  algorithm: string,
  datasetName: string,
  hyperparams: Record<string, unknown>,
  featureIndices: number[] = [0, 1],
  gridResolution: number = 50
): BoundaryResult {
  const { X, y, feature_names, target_names } = getDataset(datasetName);
  const nClasses = new Set(y).size;

  // Select 2 features
  const fi = featureIndices.slice(0, 2);
  const X_2d = X.map((row) => [row[fi[0]], row[fi[1]]]);
  const featureNames2d = [feature_names[fi[0]], feature_names[fi[1]]];

  const { scaled: X_scaled } = standardScale(X_2d);

  const xMin = Math.min(...X_scaled.map((r) => r[0])) - 1;
  const xMax = Math.max(...X_scaled.map((r) => r[0])) + 1;
  const yMin = Math.min(...X_scaled.map((r) => r[1])) - 1;
  const yMax = Math.max(...X_scaled.map((r) => r[1])) + 1;

  // Create mesh grid
  const grid_x: number[][] = [];
  const grid_y: number[][] = [];
  const step_x = (xMax - xMin) / gridResolution;
  const step_y = (yMax - yMin) / gridResolution;

  for (let i = 0; i <= gridResolution; i++) {
    const row_x: number[] = [];
    const row_y: number[] = [];
    for (let j = 0; j <= gridResolution; j++) {
      row_x.push(xMin + j * step_x);
      row_y.push(yMin + i * step_y);
    }
    grid_x.push(row_x);
    grid_y.push(row_y);
  }

  // Predict on grid using a model trained on the 2D data
  const gridPoints: Matrix = [];
  for (let i = 0; i <= gridResolution; i++) {
    for (let j = 0; j <= gridResolution; j++) {
      gridPoints.push([grid_x[i][j], grid_y[i][j]]);
    }
  }

  // Train model on full 2D data
  const trainResult = trainModel(algorithm, datasetName === 'moons' || datasetName === 'circles' || datasetName === 'classification'
    ? datasetName : datasetName, { ...hyperparams }, 0.01); // small test size since we use this for boundary

  // Actually, we need to train and predict directly
  // Let's re-train on the 2D scaled data
  const result = trainModel(algorithm, datasetName, hyperparams, 0.3);

  // For boundary, we need a separate prediction. Let's use the trained model on grid points
  // Since our trainModel function doesn't return the model, we need to retrain for boundary
  // Simpler approach: train on all 2D data, predict on grid
  const allTrainResult = trainModelForBoundary(algorithm, X_scaled, y, hyperparams, nClasses, feature_names);
  const gridZ: number[][] = [];
  let idx = 0;
  for (let i = 0; i <= gridResolution; i++) {
    const row: number[] = [];
    for (let j = 0; j <= gridResolution; j++) {
      row.push(allTrainResult[idx]);
      idx++;
    }
    gridZ.push(row);
  }

  return {
    grid_x,
    grid_y,
    grid_z: gridZ,
    points_x: X_scaled.map((r) => r[0]),
    points_y: X_scaled.map((r) => r[1]),
    points_labels: y,
    feature_names: featureNames2d,
    n_classes: nClasses,
    target_names,
  };
}

// ============================================================
// Decision Boundary for custom dataset (CSV import)
// ============================================================
export function computeDecisionBoundaryWithData(
  algorithm: string,
  X: Matrix,
  y: Label[],
  feature_names: string[],
  target_names: string[],
  hyperparams: Record<string, unknown>,
  featureIndices: number[] = [0, 1],
  gridResolution: number = 50
): BoundaryResult {
  const nClasses = new Set(y).size;

  // Select 2 features
  const fi = featureIndices.slice(0, 2);
  const safeFi = [
    Math.min(fi[0], feature_names.length - 1),
    Math.min(fi[1], feature_names.length - 1),
  ];
  const X_2d = X.map((row) => [row[safeFi[0]] ?? 0, row[safeFi[1]] ?? 0]);
  const featureNames2d = [feature_names[safeFi[0]], feature_names[safeFi[1]]];

  const { scaled: X_scaled } = standardScale(X_2d);

  const xMin = Math.min(...X_scaled.map((r) => r[0])) - 1;
  const xMax = Math.max(...X_scaled.map((r) => r[0])) + 1;
  const yMin = Math.min(...X_scaled.map((r) => r[1])) - 1;
  const yMax = Math.max(...X_scaled.map((r) => r[1])) + 1;

  const grid_x: number[][] = [];
  const grid_y: number[][] = [];
  const step_x = (xMax - xMin) / gridResolution;
  const step_y = (yMax - yMin) / gridResolution;

  for (let i = 0; i <= gridResolution; i++) {
    const row_x: number[] = [];
    const row_y: number[] = [];
    for (let j = 0; j <= gridResolution; j++) {
      row_x.push(xMin + j * step_x);
      row_y.push(yMin + i * step_y);
    }
    grid_x.push(row_x);
    grid_y.push(row_y);
  }

  const allTrainResult = trainModelForBoundary(algorithm, X_scaled, y, hyperparams, nClasses, feature_names);
  const gridZ: number[][] = [];
  let idx = 0;
  for (let i = 0; i <= gridResolution; i++) {
    const row: number[] = [];
    for (let j = 0; j <= gridResolution; j++) {
      row.push(allTrainResult[idx]);
      idx++;
    }
    gridZ.push(row);
  }

  return {
    grid_x,
    grid_y,
    grid_z: gridZ,
    points_x: X_scaled.map((r) => r[0]),
    points_y: X_scaled.map((r) => r[1]),
    points_labels: y,
    feature_names: featureNames2d,
    n_classes: nClasses,
    target_names,
  };
}

function trainModelForBoundary(algorithm: string, X: Matrix, y: Label[], hyperparams: Record<string, unknown>, nClasses: number, featureNames: string[]): Label[] {
  // Predict on the same data's grid - we need to build model and predict on grid
  // For efficiency, train on X,y and predict on X (for boundary visualization)
  const n = X.length;
  const p = X[0].length;
  const rand = seedRandom(42);
  // Create grid points (will be set externally)
  // This function returns predictions for X as a proxy

  switch (algorithm) {
    case 'knn': {
      const model = new KNNClassifier(
        Number(hyperparams.n_neighbors || 5),
        String(hyperparams.weights || 'uniform'),
        Number(hyperparams.p || 2)
      );
      model.fit(X, y);
      // We need grid predictions - create grid
      const xMin = Math.min(...X.map((r) => r[0])) - 1;
      const xMax = Math.max(...X.map((r) => r[0])) + 1;
      const yMin2 = Math.min(...X.map((r) => r[1])) - 1;
      const yMax2 = Math.max(...X.map((r) => r[1])) + 1;
      const res = 50;
      const gridPts: Matrix = [];
      for (let i = 0; i <= res; i++) {
        for (let j = 0; j <= res; j++) {
          gridPts.push([xMin + j * (xMax - xMin) / res, yMin2 + i * (yMax2 - yMin2) / res]);
        }
      }
      return model.predict(gridPts);
    }

    case 'decision_tree': {
      const maxDepth = hyperparams.max_depth == null || String(hyperparams.max_depth) === 'null'
        ? null : Number(hyperparams.max_depth);
      const tree = buildDecisionTree(X, y, maxDepth && maxDepth > 0 ? maxDepth : null,
        Number(hyperparams.min_samples_split || 2), Number(hyperparams.min_samples_leaf || 1),
        String(hyperparams.criterion || 'gini') as 'gini' | 'entropy', null
      );
      const xMin = Math.min(...X.map((r) => r[0])) - 1;
      const xMax = Math.max(...X.map((r) => r[0])) + 1;
      const yMin2 = Math.min(...X.map((r) => r[1])) - 1;
      const yMax2 = Math.max(...X.map((r) => r[1])) + 1;
      const res = 50;
      const preds: Label[] = [];
      for (let i = 0; i <= res; i++) {
        for (let j = 0; j <= res; j++) {
          preds.push(predictTree(tree, [xMin + j * (xMax - xMin) / res, yMin2 + i * (yMax2 - yMin2) / res]));
        }
      }
      return preds;
    }

    case 'random_forest': {
      const nEstimators = Number(hyperparams.n_estimators || 100);
      const maxDepth = hyperparams.max_depth == null || String(hyperparams.max_depth) === 'null'
        ? null : Number(hyperparams.max_depth);
      const maxFeatures = String(hyperparams.max_features || 'sqrt') === 'sqrt'
        ? Math.floor(Math.sqrt(p)) : Math.floor(Number(hyperparams.max_features || 1) * p) || p;
      const trees: TreeNode[] = [];
      for (let t = 0; t < nEstimators; t++) {
        const r2 = seedRandom(t + 42);
        const bootstrapIdx = Array.from({ length: n }, () => Math.floor(r2() * n));
        const bX = bootstrapIdx.map((i) => X[i]);
        const by = bootstrapIdx.map((i) => y[i]);
        trees.push(buildDecisionTree(bX, by, maxDepth && maxDepth > 0 ? maxDepth : null,
          Number(hyperparams.min_samples_split || 2), Number(hyperparams.min_samples_leaf || 1),
          'gini', maxFeatures, 0, r2
        ));
      }
      const xMin = Math.min(...X.map((r) => r[0])) - 1;
      const xMax = Math.max(...X.map((r) => r[0])) + 1;
      const yMin2 = Math.min(...X.map((r) => r[1])) - 1;
      const yMax2 = Math.max(...X.map((r) => r[1])) + 1;
      const res = 50;
      const preds: Label[] = [];
      for (let i = 0; i <= res; i++) {
        for (let j = 0; j <= res; j++) {
          const pt = [xMin + j * (xMax - xMin) / res, yMin2 + i * (yMax2 - yMin2) / res];
          preds.push(mode(trees.map((t2) => predictTree(t2, pt))));
        }
      }
      return preds;
    }

    case 'logistic_regression': {
      const model = new LogisticRegressionClassifier(Number(hyperparams.C || 1.0), Number(hyperparams.max_iter || 200));
      model.fit(X, y);
      const xMin = Math.min(...X.map((r) => r[0])) - 1;
      const xMax = Math.max(...X.map((r) => r[0])) + 1;
      const yMin2 = Math.min(...X.map((r) => r[1])) - 1;
      const yMax2 = Math.max(...X.map((r) => r[1])) + 1;
      const res = 50;
      const gridPts: Matrix = [];
      for (let i = 0; i <= res; i++) {
        for (let j = 0; j <= res; j++) {
          gridPts.push([xMin + j * (xMax - xMin) / res, yMin2 + i * (yMax2 - yMin2) / res]);
        }
      }
      return model.predict(gridPts);
    }

    case 'neural_network': {
      const hiddenSizes = String(hyperparams.hidden_layer_sizes || '100').split(',').map(Number);
      const model = new MLPClassifier(hiddenSizes, String(hyperparams.activation || 'relu'),
        Number(hyperparams.alpha || 0.0001), Number(hyperparams.learning_rate_init || 0.001),
        Number(hyperparams.max_iter || 200));
      model.fit(X, y);
      const xMin = Math.min(...X.map((r) => r[0])) - 1;
      const xMax = Math.max(...X.map((r) => r[0])) + 1;
      const yMin2 = Math.min(...X.map((r) => r[1])) - 1;
      const yMax2 = Math.max(...X.map((r) => r[1])) + 1;
      const res = 50;
      const gridPts: Matrix = [];
      for (let i = 0; i <= res; i++) {
        for (let j = 0; j <= res; j++) {
          gridPts.push([xMin + j * (xMax - xMin) / res, yMin2 + i * (yMax2 - yMin2) / res]);
        }
      }
      return model.predict(gridPts);
    }

    case 'xgboost': {
      // Simplified XGBoost boundary - use decision tree as proxy for speed
      const nEstimators = Math.min(Number(hyperparams.n_estimators || 100), 50);
      const maxDepth = Number(hyperparams.max_depth || 6);
      const learningRate = Number(hyperparams.learning_rate || 0.3);
      const yOneHot = y.map((c) => { const v = new Array(nClasses).fill(0); v[c] = 1; return v; });
      let residuals = yOneHot.map((r) => [...r]);
      const treeStore: { tree: TreeNode; classIdx: number }[] = [];
      for (let round = 0; round < nEstimators; round++) {
        for (let c = 0; c < nClasses; c++) {
          const yResid = residuals.map((r) => r[c] > 0.5 ? 1 : 0);
          const tree = buildDecisionTree(X, yResid, maxDepth, 2, 1, 'gini', null);
          treeStore.push({ tree, classIdx: c });
          for (let i = 0; i < n; i++) {
            residuals[i][c] = yOneHot[i][c] - learningRate * predictTree(tree, X[i]);
          }
        }
      }
      const xMin = Math.min(...X.map((r) => r[0])) - 1;
      const xMax = Math.max(...X.map((r) => r[0])) + 1;
      const yMin2 = Math.min(...X.map((r) => r[1])) - 1;
      const yMax2 = Math.max(...X.map((r) => r[1])) + 1;
      const res = 50;
      const preds: Label[] = [];
      for (let i = 0; i <= res; i++) {
        for (let j = 0; j <= res; j++) {
          const pt = [xMin + j * (xMax - xMin) / res, yMin2 + i * (yMax2 - yMin2) / res];
          const scores = new Array(nClasses).fill(0);
          for (const { tree, classIdx } of treeStore) scores[classIdx] += learningRate * predictTree(tree, pt);
          preds.push(scores.indexOf(Math.max(...scores)));
        }
      }
      return preds;
    }

    default:
      return X.map(() => 0);
  }
}

// ============================================================
// REGRESSION: Datasets
// ============================================================
function makeRegressionDataset(n: number, nFeatures: number, noise: number, seed: number) {
  const rand = seedRandom(seed);
  const randn = () => {
    const u1 = rand();
    const u2 = rand();
    return Math.sqrt(-2 * Math.log(u1 + 1e-10)) * Math.cos(2 * Math.PI * u2);
  };
  const featureNames = Array.from({ length: nFeatures }, (_, i) => `X${i + 1}`);
  // Generate random coefficients
  const coefs = Array.from({ length: nFeatures }, () => (rand() - 0.5) * 10);
  const bias = (rand() - 0.5) * 5;
  const X: Matrix = [];
  const y: number[] = [];
  for (let i = 0; i < n; i++) {
    const row = Array.from({ length: nFeatures }, () => randn() * 3);
    let target = bias;
    for (let j = 0; j < nFeatures; j++) target += coefs[j] * row[j];
    target += randn() * noise;
    X.push(row);
    y.push(target);
  }
  return { X, y, feature_names: featureNames, target_names: ['valor'] };
}

function getBoston() {
  const rand = seedRandom(42);
  const randn = () => {
    const u1 = rand();
    const u2 = rand();
    return Math.sqrt(-2 * Math.log(u1 + 1e-10)) * Math.cos(2 * Math.PI * u2);
  };
  const featureNames = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT'];
  const n = 506;
  const nFeatures = 13;
  // Generate synthetic boston-housing-like data
  // Key relationships: RM (rooms) positively correlated, LSTAT negatively correlated, etc.
  const X: Matrix = [];
  const y: number[] = [];
  for (let i = 0; i < n; i++) {
    const lstat = rand() * 30 + 2;
    const rm = 4 + rand() * 6;
    const crim = Math.exp(randn() * 1.5 - 1);
    const zn = rand() < 0.7 ? 0 : rand() * 100;
    const indus = 2 + rand() * 20;
    const chas = rand() < 0.93 ? 0 : 1;
    const nox = 0.35 + rand() * 0.45;
    const age = 20 + rand() * 80;
    const dis = 1 + rand() * 10;
    const rad = rand() < 0.3 ? 1 : rand() < 0.6 ? 4 : rand() < 0.8 ? 5 : 24;
    const tax = 180 + rand() * 350;
    const ptratio = 12 + rand() * 9;
    const bVal = 300 + rand() * 100;
    const row = [crim, zn, indus, chas, nox, rm, age, dis, rad, tax, ptratio, bVal, lstat];
    X.push(row);
    // Simulate price based on key features
    let price = 35 + 5 * rm - 0.6 * lstat - 0.02 * crim - 0.01 * age + 0.01 * dis - 0.03 * nox * 10;
    price += randn() * 3;
    price = Math.max(5, price);
    y.push(round4(price));
  }
  return { X, y, feature_names: featureNames, target_names: ['MEDV'] };
}

// ============================================================
// REGRESSION: Metrics
// ============================================================
export function computeRegressionMetrics(yTrue: number[], yPred: number[]): { mse: number; rmse: number; mae: number; r2: number } {
  const n = yTrue.length;
  let sumSqErr = 0;
  let sumAbsErr = 0;
  let sumSqTotal = 0;
  const mean = yTrue.reduce((a, b) => a + b, 0) / n;
  for (let i = 0; i < n; i++) {
    const diff = yTrue[i] - yPred[i];
    sumSqErr += diff * diff;
    sumAbsErr += Math.abs(diff);
    sumSqTotal += (yTrue[i] - mean) ** 2;
  }
  const mse = sumSqErr / n;
  const rmse = Math.sqrt(mse);
  const mae = sumAbsErr / n;
  const r2 = sumSqTotal > 0 ? 1 - sumSqErr / sumSqTotal : 0;
  return { mse: round4(mse), rmse: round4(rmse), mae: round4(mae), r2: round4(r2) };
}

// ============================================================
// REGRESSION: Train/Test Split (non-stratified, for continuous y)
// ============================================================
function trainTestSplitRegression(X: Matrix, y: number[], testSize: number, seed: number) {
  const n = X.length;
  const indices = shuffleArray(Array.from({ length: n }, (_, i) => i), seed);
  const splitIdx = Math.floor(n * (1 - testSize));
  const trainIdx = indices.slice(0, splitIdx);
  const testIdx = indices.slice(splitIdx);
  return {
    X_train: trainIdx.map((i) => X[i]),
    X_test: testIdx.map((i) => X[i]),
    y_train: trainIdx.map((i) => y[i]),
    y_test: testIdx.map((i) => y[i]),
  };
}

// ============================================================
// REGRESSION: Regression Tree (variance/MSE instead of Gini)
// ============================================================
function mseImpurity(y: number[]): number {
  const n = y.length;
  if (n === 0) return 0;
  const mean = y.reduce((a, b) => a + b, 0) / n;
  let sum = 0;
  for (const v of y) sum += (v - mean) ** 2;
  return sum / n;
}

function buildRegressionTree(
  X: Matrix, y: number[], maxDepth: number | null, minSamplesSplit: number,
  minSamplesLeaf: number, nFeatures: number | null,
  depth: number = 0, rand?: () => number
): TreeNode {
  const nSamples = y.length;
  const meanVal = y.reduce((a, b) => a + b, 0) / nSamples;

  if (nSamples <= minSamplesSplit || (maxDepth !== null && depth >= maxDepth) || nSamples < 2) {
    return { label: round4(meanVal), samples: nSamples, impurity: mseImpurity(y) };
  }

  const nFeaturesTotal = X[0].length;
  const featureIndices = nFeatures && nFeatures < nFeaturesTotal
    ? shuffleArray(Array.from({ length: nFeaturesTotal }, (_, i) => i), depth * 1000 + nSamples).slice(0, nFeatures)
    : Array.from({ length: nFeaturesTotal }, (_, i) => i);

  let bestFeature = -1;
  let bestThreshold = 0;
  let bestImpurity = Infinity;
  let bestLeftIdx: number[] = [];
  let bestRightIdx: number[] = [];
  const parentMse = mseImpurity(y);

  for (const fi of featureIndices) {
    const values = [...new Set(X.map((row) => round4(row[fi] * 100) / 100))].sort((a, b) => a - b);
    // Sample thresholds for efficiency
    const thresholds = values.length > 50
      ? Array.from({ length: 50 }, (_, k) => values[Math.floor(k * values.length / 50)])
      : values;
    for (let t = 0; t < thresholds.length - 1; t++) {
      const threshold = (thresholds[t] + thresholds[t + 1]) / 2;
      const leftIdx: number[] = [];
      const rightIdx: number[] = [];
      for (let i = 0; i < nSamples; i++) {
        if (X[i][fi] <= threshold) leftIdx.push(i);
        else rightIdx.push(i);
      }
      if (leftIdx.length < minSamplesLeaf || rightIdx.length < minSamplesLeaf) continue;
      const leftY = leftIdx.map((i) => y[i]);
      const rightY = rightIdx.map((i) => y[i]);
      const weightedMse = (leftY.length * mseImpurity(leftY) + rightY.length * mseImpurity(rightY)) / nSamples;
      if (weightedMse < bestImpurity) {
        bestImpurity = weightedMse;
        bestFeature = fi;
        bestThreshold = threshold;
        bestLeftIdx = leftIdx;
        bestRightIdx = rightIdx;
      }
    }
  }

  if (bestFeature === -1) return { label: round4(meanVal), samples: nSamples, impurity: parentMse };

  const left = buildRegressionTree(
    bestLeftIdx.map((i) => X[i]), bestLeftIdx.map((i) => y[i]),
    maxDepth, minSamplesSplit, minSamplesLeaf, nFeatures, depth + 1, rand
  );
  const right = buildRegressionTree(
    bestRightIdx.map((i) => X[i]), bestRightIdx.map((i) => y[i]),
    maxDepth, minSamplesSplit, minSamplesLeaf, nFeatures, depth + 1, rand
  );

  return { feature: bestFeature, threshold: bestThreshold, left, right, samples: nSamples, impurity: parentMse };
}

function regressionTreeToText(node: TreeNode, featureNames: string[], prefix: string = ''): string {
  if (node.label !== undefined) return `${prefix}|--- valor: ${node.label.toFixed(2)}\n`;
  const fname = featureNames[node.feature!] || `Feature ${node.feature}`;
  let text = `${prefix}|--- ${fname} <= ${node.threshold!.toFixed(2)}\n`;
  text += regressionTreeToText(node.left!, featureNames, prefix + '|   ');
  text += `${prefix}|--- ${fname} >  ${node.threshold!.toFixed(2)}\n`;
  text += regressionTreeToText(node.right!, featureNames, prefix + '|   ');
  return text;
}

function regressionTreeFeatureImportance(node: TreeNode, nFeatures: number): number[] {
  const importance = new Array(nFeatures).fill(0);
  function traverse(n: TreeNode) {
    if (n.feature !== undefined) {
      const leftSize = n.left?.samples || 0;
      const rightSize = n.right?.samples || 0;
      const leftImp = n.left?.impurity || 0;
      const rightImp = n.right?.impurity || 0;
      const currentImp = n.impurity || 0;
      // Importance = reduction in impurity * number of samples
      importance[n.feature] += (leftSize + rightSize) * (currentImp - (leftSize * leftImp + rightSize * rightImp) / (leftSize + rightSize));
      traverse(n.left!);
      traverse(n.right!);
    }
  }
  traverse(node);
  const total = importance.reduce((a, b) => a + b, 0);
  if (total <= 0) return importance.map(() => 1 / nFeatures);
  return importance.map((v) => v / total);
}

// ============================================================
// REGRESSION: KNN Regressor
// ============================================================
class KNNRegressor {
  private X: Matrix = [];
  private y: number[] = [];

  constructor(private k: number = 5, private weights: string = 'uniform', private p: number = 2) {}

  fit(X: Matrix, y: number[]) { this.X = X; this.y = y; }

  predict(X: Matrix): number[] {
    return X.map((x) => this.predictOne(x));
  }

  private predictOne(x: Vector): number {
    const distFn = this.p === 1 ? manhattan : euclidean;
    const distances = this.X.map((trainX, i) => ({ dist: distFn(x, trainX), value: this.y[i] }));
    distances.sort((a, b) => a.dist - b.dist);
    const neighbors = distances.slice(0, this.k);
    if (this.weights === 'distance') {
      let weightedSum = 0;
      let weightTotal = 0;
      for (const n of neighbors) {
        const w = 1 / (n.dist + 1e-10);
        weightedSum += w * n.value;
        weightTotal += w;
      }
      return weightedSum / weightTotal;
    }
    return neighbors.reduce((s, n) => s + n.value, 0) / neighbors.length;
  }
}

// ============================================================
// REGRESSION: Linear Regression (SGD with L2)
// ============================================================
class LinearRegression {
  private weights: Vector = [];
  private bias: number = 0;

  constructor(private alpha: number = 0.0001, private maxIter: number = 200, private learningRate: number = 0.01) {}

  fit(X: Matrix, y: number[]) {
    const n = X.length;
    const p = X[0].length;
    this.weights = new Array(p).fill(0);
    this.bias = 0;

    for (let iter = 0; iter < this.maxIter; iter++) {
      for (let i = 0; i < n; i++) {
        let pred = this.bias;
        for (let j = 0; j < p; j++) pred += this.weights[j] * X[i][j];
        const error = y[i] - pred;
        for (let j = 0; j < p; j++) {
          this.weights[j] += this.learningRate * (error * X[i][j] - this.alpha * this.weights[j]);
        }
        this.bias += this.learningRate * error;
      }
    }
  }

  predict(X: Matrix): number[] {
    return X.map((x) => {
      let pred = this.bias;
      for (let j = 0; j < x.length; j++) pred += this.weights[j] * x[j];
      return pred;
    });
  }

  getFeatureImportance(): number[] {
    return this.weights.map(Math.abs);
  }
}

// ============================================================
// REGRESSION: MLP Regressor
// ============================================================
class MLPRegressor {
  private layerWeights: Matrix[] = [];
  private layerBiases: Vector[] = [];
  private hiddenSizes: number[];

  constructor(
    hiddenLayerSizes: number[] = [100],
    private activation: string = 'relu',
    private alpha: number = 0.0001,
    private learningRate: number = 0.001,
    private maxIter: number = 200
  ) {
    this.hiddenSizes = hiddenLayerSizes;
  }

  fit(X: Matrix, y: number[]) {
    const n = X.length;
    const p = X[0].length;
    // Output layer has 1 neuron for regression
    const layerSizes = [p, ...this.hiddenSizes, 1];
    const rand = seedRandom(42);

    this.layerWeights = [];
    this.layerBiases = [];
    for (let l = 0; l < layerSizes.length - 1; l++) {
      const fan = layerSizes[l];
      const scale = Math.sqrt(2 / fan);
      this.layerWeights.push(Array.from({ length: layerSizes[l + 1] }, () =>
        Array.from({ length: layerSizes[l] }, () => (rand() - 0.5) * 2 * scale)
      ));
      this.layerBiases.push(new Array(layerSizes[l + 1]).fill(0));
    }

    const activationFn = this.activation === 'tanh' ? tanh : this.activation === 'logistic' ? sigmoid : relu;
    const activationDeriv = this.activation === 'tanh'
      ? (x: number) => 1 - Math.tanh(x) ** 2
      : this.activation === 'logistic'
        ? (x: number) => { const s = sigmoid(x); return s * (1 - s); }
        : (x: number) => x > 0 ? 1 : 0;

    for (let iter = 0; iter < this.maxIter; iter++) {
      for (let i = 0; i < n; i++) {
        // Forward pass
        const activations: Matrix = [X[i]];
        const preActivations: Matrix = [];
        for (let l = 0; l < this.layerWeights.length; l++) {
          const input = activations[l];
          const pre: Vector = [];
          for (let j = 0; j < this.layerWeights[l].length; j++) {
            let sum = this.layerBiases[l][j];
            for (let k = 0; k < input.length; k++) sum += this.layerWeights[l][j][k] * input[k];
            pre.push(sum);
          }
          preActivations.push(pre);
          if (l < this.layerWeights.length - 1) {
            activations.push(pre.map(activationFn));
          } else {
            // Linear output for regression (no activation)
            activations.push([...pre]);
          }
        }

        // Output error: MSE gradient = pred - true
        const output = activations[activations.length - 1];
        const errors: Vector = output.map((o) => y[i] - o);

        // Backward pass
        const deltas: Vector[] = [errors];
        for (let l = this.layerWeights.length - 2; l >= 0; l--) {
          const delta: Vector = [];
          for (let j = 0; j < this.layerWeights[l].length; j++) {
            let sum = 0;
            for (let k = 0; k < this.layerWeights[l + 1].length; k++) {
              sum += this.layerWeights[l + 1][k][j] * deltas[0][k];
            }
            delta.push(sum * activationDeriv(preActivations[l][j]));
          }
          deltas.unshift(delta);
        }

        // Update weights
        for (let l = 0; l < this.layerWeights.length; l++) {
          for (let j = 0; j < this.layerWeights[l].length; j++) {
            for (let k = 0; k < this.layerWeights[l][j].length; k++) {
              this.layerWeights[l][j][k] += this.learningRate * (deltas[l][j] * activations[l][k] - this.alpha * this.layerWeights[l][j][k]);
            }
            this.layerBiases[l][j] += this.learningRate * deltas[l][j];
          }
        }
      }
    }
  }

  predict(X: Matrix): number[] {
    const activationFn = this.activation === 'tanh' ? tanh : this.activation === 'logistic' ? sigmoid : relu;
    return X.map((x) => {
      let current = x;
      for (let l = 0; l < this.layerWeights.length; l++) {
        const next: Vector = [];
        for (let j = 0; j < this.layerWeights[l].length; j++) {
          let sum = this.layerBiases[l][j];
          for (let k = 0; k < current.length; k++) sum += this.layerWeights[l][j][k] * current[k];
          if (l < this.layerWeights.length - 1) next.push(activationFn(sum));
          else next.push(sum); // Linear output
        }
        current = next;
      }
      return current[0];
    });
  }
}

// ============================================================
// REGRESSION: Main training function
// ============================================================
export function trainRegressionModel(
  algorithm: string,
  datasetName: string,
  hyperparams: Record<string, unknown>,
  testSize: number = 0.3
): RegressionTrainResult {
  try {
    const { X, y, feature_names, target_names } = getDataset(datasetName);
    return trainRegressionModelWithData(algorithm, X, y, feature_names, target_names, hyperparams, testSize);
  } catch (error: unknown) {
    return {
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
      error: error instanceof Error ? error.message : 'Error desconocido',
    };
  }
}

export function trainRegressionModelWithData(
  algorithm: string,
  X: Matrix,
  y: number[],
  feature_names: string[],
  target_names: string[],
  hyperparams: Record<string, unknown>,
  testSize: number = 0.3
): RegressionTrainResult {
  try {
    const { X_train, X_test, y_train, y_test } = trainTestSplitRegression(X, y, testSize, 42);
    const { scaled: X_train_s, means, stds } = standardScale(X_train);
    const X_test_s = scaleWith(X_test, means, stds);

    let y_pred: number[];
    let featureImportance: { feature: string; importance: number; index: number }[] = [];
    let treeText: string | null = null;

    switch (algorithm) {
      case 'knn_regressor': {
        const model = new KNNRegressor(
          Number(hyperparams.n_neighbors || 5),
          String(hyperparams.weights || 'uniform'),
          Number(hyperparams.p || 2)
        );
        model.fit(X_train_s, y_train);
        y_pred = model.predict(X_test_s);
        break;
      }

      case 'linear_regression': {
        const model = new LinearRegression(
          Number(hyperparams.alpha || 0.0001),
          Number(hyperparams.max_iter || 200),
          Number(hyperparams.learning_rate || 0.01)
        );
        model.fit(X_train_s, y_train);
        y_pred = model.predict(X_test_s);
        const imp = model.getFeatureImportance();
        const impTotal = imp.reduce((a, b) => a + b, 0) || 1;
        featureImportance = feature_names.map((f, i) => ({ feature: f, importance: round4(imp[i] / impTotal), index: i }))
          .sort((a, b) => b.importance - a.importance);
        break;
      }

      case 'mlp_regressor': {
        const hiddenSizes = String(hyperparams.hidden_layer_sizes || '100').split(',').map(Number);
        const model = new MLPRegressor(
          hiddenSizes,
          String(hyperparams.activation || 'relu'),
          Number(hyperparams.alpha || 0.0001),
          Number(hyperparams.learning_rate_init || 0.001),
          Number(hyperparams.max_iter || 200)
        );
        model.fit(X_train_s, y_train);
        y_pred = model.predict(X_test_s);
        break;
      }

      case 'decision_tree_regressor': {
        const maxDepth = hyperparams.max_depth == null || hyperparams.max_depth === null || String(hyperparams.max_depth) === 'null'
          ? null : Number(hyperparams.max_depth);
        const tree = buildRegressionTree(
          X_train_s, y_train, maxDepth && maxDepth > 0 ? maxDepth : null,
          Number(hyperparams.min_samples_split || 2),
          Number(hyperparams.min_samples_leaf || 1),
          null
        );
        y_pred = X_test_s.map((x) => predictTree(tree, x));
        const imp = regressionTreeFeatureImportance(tree, feature_names.length);
        featureImportance = feature_names.map((f, i) => ({ feature: f, importance: round4(imp[i]), index: i }))
          .sort((a, b) => b.importance - a.importance);
        treeText = regressionTreeToText(tree, feature_names);
        break;
      }

      case 'random_forest_regressor': {
        const nEstimators = Number(hyperparams.n_estimators || 100);
        const maxDepth = hyperparams.max_depth == null || String(hyperparams.max_depth) === 'null'
          ? null : Number(hyperparams.max_depth);
        const maxFeatures = String(hyperparams.max_features || 'sqrt') === 'sqrt'
          ? Math.floor(Math.sqrt(feature_names.length))
          : String(hyperparams.max_features) === 'log2'
            ? Math.floor(Math.log2(feature_names.length))
            : Math.floor(Number(hyperparams.max_features || 1) * feature_names.length) || feature_names.length;
        const allPredictions: number[][] = [];
        const impAvg = new Array(feature_names.length).fill(0);
        for (let t = 0; t < nEstimators; t++) {
          const rand = seedRandom(t + 42);
          const bootstrapIdx = Array.from({ length: X_train_s.length }, () => Math.floor(rand() * X_train_s.length));
          const bX = bootstrapIdx.map((i) => X_train_s[i]);
          const by = bootstrapIdx.map((i) => y_train[i]);
          const tree = buildRegressionTree(bX, by, maxDepth && maxDepth > 0 ? maxDepth : null,
            Number(hyperparams.min_samples_split || 2), Number(hyperparams.min_samples_leaf || 1),
            maxFeatures, 0, rand
          );
          allPredictions.push(X_test_s.map((x) => predictTree(tree, x)));
          const imp = regressionTreeFeatureImportance(tree, feature_names.length);
          for (let j = 0; j < imp.length; j++) impAvg[j] += imp[j];
        }
        // Average predictions for regression
        y_pred = X_test_s.map((_, i) => {
          const sum = allPredictions.reduce((s, p) => s + p[i], 0);
          return sum / allPredictions.length;
        });
        for (let j = 0; j < impAvg.length; j++) impAvg[j] /= nEstimators;
        featureImportance = feature_names.map((f, i) => ({ feature: f, importance: round4(impAvg[i]), index: i }))
          .sort((a, b) => b.importance - a.importance);
        break;
      }

      case 'xgboost_regressor': {
        const nEstimators = Number(hyperparams.n_estimators || 100);
        const maxDepth = Number(hyperparams.max_depth || 6);
        const learningRate = Number(hyperparams.learning_rate || 0.3);
        const nTrain = X_train_s.length;
        const nFeatures = X_train_s[0].length;

        // Gradient boosting with actual continuous residuals
        let predictions = new Array(nTrain).fill(0);
        const trees: TreeNode[] = [];
        const impAvg = new Array(nFeatures).fill(0);

        for (let round = 0; round < nEstimators; round++) {
          // Compute residuals (actual - predicted)
          const residuals = y_train.map((yi, i) => yi - predictions[i]);
          const tree = buildRegressionTree(X_train_s, residuals, maxDepth, 2, 1, null);
          trees.push(tree);
          const imp = regressionTreeFeatureImportance(tree, nFeatures);
          for (let j = 0; j < imp.length; j++) impAvg[j] += imp[j];
          // Update predictions
          const treePreds = X_train_s.map((x) => predictTree(tree, x));
          for (let i = 0; i < nTrain; i++) {
            predictions[i] += learningRate * treePreds[i];
          }
        }

        // Predict on test set
        y_pred = X_test_s.map((x) => {
          let pred = 0;
          for (const tree of trees) {
            pred += learningRate * predictTree(tree, x);
          }
          return pred;
        });

        for (let j = 0; j < impAvg.length; j++) impAvg[j] /= nEstimators || 1;
        featureImportance = feature_names.map((f, i) => ({ feature: f, importance: round4(impAvg[i]), index: i }))
          .sort((a, b) => b.importance - a.importance);
        break;
      }

      default:
        throw new Error(`Algoritmo de regresión desconocido: ${algorithm}`);
    }

    const metrics = computeRegressionMetrics(y_test, y_pred);

    return {
      success: true,
      metrics,
      feature_importance: featureImportance,
      tree_text: treeText,
      n_train: X_train.length,
      n_test: X_test.length,
      target_names,
      feature_names,
      y_test,
      y_pred,
    };
  } catch (error: unknown) {
    return {
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
      error: error instanceof Error ? error.message : 'Error desconocido',
    };
  }
}

// ============================================================
// REGRESSION: Plot data for scatter + regression line
// ============================================================
export function computeRegressionPlotData(
  algorithm: string,
  datasetName: string,
  hyperparams: Record<string, unknown>,
  featureIndices: number[] = [0, 1]
): RegressionPlotData {
  const { X, y, feature_names } = getDataset(datasetName);
  return computeRegressionPlotDataWithData(algorithm, X, y, feature_names, hyperparams, featureIndices);
}

export function computeRegressionPlotDataWithData(
  algorithm: string,
  X: Matrix,
  y: number[],
  feature_names: string[],
  hyperparams: Record<string, unknown>,
  featureIndices: number[] = [0, 1]
): RegressionPlotData {
  const fi0 = Math.min(featureIndices[0] || 0, feature_names.length - 1);
  const fi1 = Math.min(featureIndices[1] || Math.min(1, feature_names.length - 1), feature_names.length - 1);
  // Use feature fi0 for x-axis, y for target
  const { scaled: X_s, means, stds } = standardScale(X);
  const xVals = X_s.map((r) => r[fi0]);

  // Train a regression model on all data
  const y_pred = trainRegressionPredictAll(algorithm, X_s, y, hyperparams);

  // Build regression line (sorted by x)
  const paired = xVals.map((x, i) => ({ x, pred: y_pred[i] })).sort((a, b) => a.x - b.x);
  const lineX = paired.map((p) => p.x);
  const lineY = paired.map((p) => p.pred);

  return {
    points_x: xVals,
    points_y: X_s.map((r) => r[fi1]),
    points_true: y,
    points_pred: y_pred,
    feature_names: [feature_names[fi0], feature_names[fi1]],
    regression_line_x: lineX,
    regression_line_y: lineY,
  };
}

function trainRegressionPredictAll(
  algorithm: string,
  X: Matrix,
  y: number[],
  hyperparams: Record<string, unknown>
): number[] {
  const n = X.length;
  const p = X[0].length;
  const featureNames = Array.from({ length: p }, (_, i) => `F${i}`);

  switch (algorithm) {
    case 'knn_regressor': {
      const model = new KNNRegressor(
        Number(hyperparams.n_neighbors || 5),
        String(hyperparams.weights || 'uniform'),
        Number(hyperparams.p || 2)
      );
      model.fit(X, y);
      return model.predict(X);
    }
    case 'linear_regression': {
      const model = new LinearRegression(
        Number(hyperparams.alpha || 0.0001),
        Number(hyperparams.max_iter || 200),
        Number(hyperparams.learning_rate || 0.01)
      );
      model.fit(X, y);
      return model.predict(X);
    }
    case 'mlp_regressor': {
      const hiddenSizes = String(hyperparams.hidden_layer_sizes || '100').split(',').map(Number);
      const model = new MLPRegressor(
        hiddenSizes,
        String(hyperparams.activation || 'relu'),
        Number(hyperparams.alpha || 0.0001),
        Number(hyperparams.learning_rate_init || 0.001),
        Number(hyperparams.max_iter || 200)
      );
      model.fit(X, y);
      return model.predict(X);
    }
    case 'decision_tree_regressor': {
      const maxDepth = hyperparams.max_depth == null || String(hyperparams.max_depth) === 'null'
        ? null : Number(hyperparams.max_depth);
      const tree = buildRegressionTree(
        X, y, maxDepth && maxDepth > 0 ? maxDepth : null,
        Number(hyperparams.min_samples_split || 2),
        Number(hyperparams.min_samples_leaf || 1),
        null
      );
      return X.map((x) => predictTree(tree, x));
    }
    case 'random_forest_regressor': {
      const nEstimators = Math.min(Number(hyperparams.n_estimators || 100), 30);
      const maxDepth = hyperparams.max_depth == null || String(hyperparams.max_depth) === 'null'
        ? null : Number(hyperparams.max_depth);
      const maxFeatures = String(hyperparams.max_features || 'sqrt') === 'sqrt'
        ? Math.floor(Math.sqrt(p))
        : Math.floor(Number(hyperparams.max_features || 1) * p) || p;
      const allPreds: number[][] = [];
      for (let t = 0; t < nEstimators; t++) {
        const rand = seedRandom(t + 42);
        const bootstrapIdx = Array.from({ length: n }, () => Math.floor(rand() * n));
        const bX = bootstrapIdx.map((i) => X[i]);
        const by = bootstrapIdx.map((i) => y[i]);
        const tree = buildRegressionTree(bX, by, maxDepth && maxDepth > 0 ? maxDepth : null,
          Number(hyperparams.min_samples_split || 2), Number(hyperparams.min_samples_leaf || 1),
          maxFeatures, 0, rand
        );
        allPreds.push(X.map((x) => predictTree(tree, x)));
      }
      return X.map((_, i) => allPreds.reduce((s, p) => s + p[i], 0) / allPreds.length);
    }
    case 'xgboost_regressor': {
      const nEstimators = Math.min(Number(hyperparams.n_estimators || 100), 50);
      const maxDepth = Number(hyperparams.max_depth || 6);
      const learningRate = Number(hyperparams.learning_rate || 0.3);
      let preds = new Array(n).fill(0);
      for (let round = 0; round < nEstimators; round++) {
        const residuals = y.map((yi, i) => yi - preds[i]);
        const tree = buildRegressionTree(X, residuals, maxDepth, 2, 1, null);
        const treePreds = X.map((x) => predictTree(tree, x));
        for (let i = 0; i < n; i++) preds[i] += learningRate * treePreds[i];
      }
      return preds;
    }
    default:
      return new Array(n).fill(0);
  }
}
