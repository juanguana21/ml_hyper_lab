# 🧠 ML HyperLab — Guía de Instalación para Estudiantes

## Laboratorio Interactivo de Machine Learning

---

## 📋 Requisitos Previos

Antes de instalar ML HyperLab, necesitas tener estos programas en tu computadora:

| Programa | Versión mínima | Descarga | Cómo verificar |
|---|---|---|---|
| **Node.js** | 18+ | https://nodejs.org/ (elegir versión LTS) | Abrir terminal y escribir: `node --version` |
| **Python 3** | 3.9+ | https://www.python.org/downloads/ | En terminal: `python3 --version` |
| **pip** | Viene con Python | Se instala con Python | En terminal: `pip3 --version` |

### ⚠️ Importante al instalar Python en Windows:
- Marca la casilla **"Add Python to PATH"** durante la instalación
- Si ya tienes Python pero `python3` no funciona, usa `python` en su lugar

---

## 🚀 Opción A: Instalación Rápida (Recomendada)

### Paso 1: Descargar el proyecto

Descarga la carpeta del proyecto y descomprímela en tu computadora.

### Paso 2: Abrir la terminal

- **Windows**: Abre **PowerShell** o **CMD**
- **Mac**: Abre **Terminal** (Cmd + Espacio → "Terminal")
- **Linux**: Abre tu terminal favorita

### Paso 3: Navegar a la carpeta del proyecto

```bash
cd ruta/donde/descomprimiste/ml-hyperlab
```

### Paso 4: Instalar dependencias

```bash
# Instalar dependencias de Node.js
npm install

# Instalar librerías de Python para Machine Learning
pip3 install scikit-learn xgboost numpy pandas
```

> **Nota para Windows**: Si `pip3` no funciona, usa `pip` en su lugar.

### Paso 5: ¡Ejecutar la app!

```bash
npm run dev
```

### Paso 6: Abrir en el navegador

Abre tu navegador y ve a:

```
http://localhost:3000
```

¡Listo! Ya puedes experimentar con los algoritmos de Machine Learning 🎉

---

## 🔧 Opción B: Instalación desde Cero

Si prefieres crear todo paso a paso:

### 1. Crear el proyecto Next.js

```bash
npx create-next-app@latest ml-hyperlab --typescript --tailwind --eslint --app --src-dir
```

Responde las preguntas así:
- ✔ Would you like to customize the import alias? → **No**

### 2. Entrar al proyecto

```bash
cd ml-hyperlab
```

### 3. Inicializar shadcn/ui

```bash
npx shadcn@latest init
```

Selecciona:
- Style: **New York**
- Base color: **Neutral**
- CSS variables: **Yes**

### 4. Instalar componentes de UI

```bash
npx shadcn@latest add card button badge slider select tooltip tabs separator scroll-area
```

### 5. Instalar iconos

```bash
npm install lucide-react
```

### 6. Instalar Python ML

```bash
pip3 install scikit-learn xgboost numpy pandas
```

### 7. Copiar archivos fuente

Copia los siguientes archivos del proyecto original a tu proyecto:

```
Tu proyecto debe quedar con esta estructura:

ml-hyperlab/
├── src/
│   ├── app/
│   │   ├── page.tsx                        ← Interfaz principal
│   │   ├── layout.tsx                      ← Configuración de la página
│   │   ├── globals.css                     ← Estilos (ya existe)
│   │   └── api/ml/
│   │       ├── train/route.ts              ← API de entrenamiento
│   │       ├── boundary/route.ts           ← API de frontera de decisión
│   │       └── datasets/route.ts           ← API de datasets
│   ├── lib/
│   │   ├── ml-config.ts                    ← Configuración de algoritmos
│   │   └── utils.ts                        ← Utilidades (ya existe)
│   ├── components/ui/                      ← Componentes (ya existen)
│   └── hooks/
├── ml-scripts/
│   ├── train.py                            ← Script ML de entrenamiento
│   └── boundary.py                         ← Script ML de frontera
├── package.json
└── ...
```

### 8. Ejecutar

```bash
npm run dev
```

---

## 🛑 Solución de Problemas

### ❌ "node no se reconoce como un comando"
**Problema**: Node.js no está instalado o no está en el PATH.
**Solución**:
1. Instala Node.js desde https://nodejs.org/
2. Reinicia la terminal después de instalar
3. Verifica con `node --version`

### ❌ "python3 no se reconoce como un comando"
**Problema**: Python no está en el PATH (común en Windows).
**Solución**:
1. En Windows, usa `python` en lugar de `python3`
2. Si usas Windows y Python está instalado, busca "Python" en el menú inicio y ejecuta "Add Python to PATH"
3. Reinicia la terminal

### ❌ "pip3 no se reconoce como un comando"
**Solución**: Usa `pip` en lugar de `pip3`, o prueba:
```bash
python3 -m pip install scikit-learn xgboost numpy pandas
```

### ❌ Error al instalar xgboost
**Solución en Windows**:
```bash
pip install xgboost --no-cache-dir
```
Si sigue fallando, intenta:
```bash
conda install -c conda-forge xgboost
```

### ❌ "npm run dev" falla con error de puerto
**Problema**: El puerto 3000 ya está en uso.
**Solución**: Mata el proceso anterior:
- **Mac/Linux**: `lsof -ti:3000 | xkill`
- **Windows**: En PowerShell: `Stop-Process -Id (Get-NetTCPConnection -LocalPort 3000).OwningProcess -Force`
- O simplemente reinicia la computadora

### ❌ La página se ve pero al entrenar sale error
**Problema**: Los scripts de Python no se encuentran.
**Solución**:
1. Verifica que la carpeta `ml-scripts/` esté en la raíz del proyecto
2. Verifica que Python y las librerías estén instaladas:
```bash
python3 -c "import sklearn, xgboost; print('OK')"
```

### ❌ Error "EPERM" o "EACCES" al instalar paquetes
**Problema**: Permisos insuficientes.
**Solución en Mac/Linux**:
```bash
sudo npm install
sudo pip3 install scikit-learn xgboost numpy pandas
```
**Solución en Windows**: Abre PowerShell como Administrador

---

## 📱 Cómo usar ML HyperLab

### Para el estudiante:

1. **Elige un algoritmo** → Haz clic en uno de los 6 algoritmos (XGBoost, Random Forest, KNN, etc.)
2. **Lee la explicación** → Haz clic en "Explicación completa" para entender el algoritmo
3. **Selecciona un dataset** → Te recomendamos empezar con "Two Moons" 🌙 para ver las fronteras 2D
4. **Ajusta los hiperparámetros** → Mueve los sliders y observa:
   - Cada parámetro tiene un ícono ℹ️ con explicación detallada
   - Los badges "Impacto alto/medio/bajo" te indican qué tan sensible es
5. **Presiona "Entrenar Modelo"** → Observa los resultados:
   - 📊 Métricas (Accuracy, Precision, Recall, F1)
   - 🎨 Frontera de decisión visual
   - 📋 Matriz de confusión
   - 📈 Importancia de features
6. **Experimenta** → Cambia un hiperparámetro y entrena de nuevo. ¡Observa la diferencia!

### Ejercicios sugeridos:

| Ejercicio | Algoritmo | Dataset | Qué hacer |
|---|---|---|---|
| Sobreajuste en KNN | KNN | Moons | Prueba k=1 vs k=50. ¿Qué pasa? |
| Profundidad del árbol | Decision Tree | Circles | Prueba max_depth=1 vs 20. ¿Cuándo sobreajusta? |
| Learning rate | XGBoost | Iris | Prueba lr=0.01 vs lr=1.0. ¿Qué observas? |
| Regularización | Regresión Logística | Circles | ¿Puede una línea separar círculos? |
| Capas ocultas | Red Neuronal | Circles | Prueba [50] vs [100,50,25]. ¿Mejora? |
| Número de árboles | Random Forest | Wine | Prueba 10 vs 500 árboles. ¿Cuánto cambia? |

---

## 📞 Soporte

Si tienes problemas, verifica:
1. ✅ Node.js está instalado (`node --version`)
2. ✅ Python3 está instalado (`python3 --version`)
3. ✅ Las librerías ML están instaladas (`python3 -c "import sklearn, xgboost"`)
4. ✅ Estás en la carpeta correcta del proyecto
5. ✅ Ejecutaste `npm install` antes de `npm run dev`
