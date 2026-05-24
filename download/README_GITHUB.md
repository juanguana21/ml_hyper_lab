# 🧠 ML HyperLab

**Laboratorio Interactivo de Machine Learning**

Aplicación web 100% en el navegador para experimentar con hiperparámetros de algoritmos de Machine Learning. Sin instalación, sin backend, sin Python — solo abre el enlace y experimenta.

## 🚀 Demo en vivo

Cuando actives GitHub Pages, tu app estará en:

```
https://<tu-usuario>.github.io/ml-hyperlab/
```

## ✨ Características

- 🚀 **6 algoritmos**: XGBoost, Random Forest, KNN, Red Neuronal, Árbol de Decisión, Regresión Logística
- 📊 **6 datasets**: Iris, Wine, Breast Cancer, Two Moons, Circles, Synthetic
- 🎛️ **Hiperparámetros interactivos** con sliders y explicaciones detalladas
- 📈 **Visualizaciones**: frontera de decisión 2D, matriz de confusión, importancia de features
- 📱 **Responsive**: funciona en laptop, tablet y celular
- 🌐 **Sin backend**: todo corre en el navegador con JavaScript puro
- 🇪🇸 **100% en español** con explicaciones didácticas

## 🛠️ Despliegue en GitHub Pages

### Paso 1: Crear repositorio en GitHub

```bash
# Crear repo nuevo en github.com y luego:
git init
git add .
git commit -m "ML HyperLab - primera versión"
git branch -M main
git remote add origin https://github.com/<tu-usuario>/ml-hyperlab.git
git push -u origin main
```

### Paso 2: Activar GitHub Pages

1. Ve a tu repositorio en GitHub
2. **Settings** → **Pages**
3. En "Source", selecciona **GitHub Actions**
4. El workflow se ejecutará automáticamente con cada push

### Paso 3: ¡Listo!

Tu app estará disponible en `https://<tu-usuario>.github.io/ml-hyperlab/`

## 💻 Ejecutar localmente

```bash
# Clonar
git clone https://github.com/<tu-usuario>/ml-hyperlab.git
cd ml-hyperlab

# Instalar
npm install

# Ejecutar
npm run dev

# Abrir: http://localhost:3000
```

**No necesitas Python** — los algoritmos corren 100% en JavaScript/TypeScript.

## 📚 Para el aula

### Ejercicios sugeridos

| Ejercicio | Algoritmo | Dataset | Qué hacer |
|---|---|---|---|
| Sobreajuste en KNN | KNN | Two Moons | Prueba k=1 vs k=50 |
| Profundidad del árbol | Decision Tree | Circles | Prueba max_depth=1 vs 20 |
| Learning rate | XGBoost | Iris | Prueba lr=0.01 vs lr=1.0 |
| Clasificador lineal | Regresión Logística | Circles | ¿Puede separar círculos? |
| Capas ocultas | Red Neuronal | Circles | Prueba [50] vs [100,50,25] |
| Número de árboles | Random Forest | Wine | Prueba 10 vs 500 árboles |

## 🏗️ Arquitectura

- **Frontend**: Next.js 16 + TypeScript + Tailwind CSS + shadcn/ui
- **ML Engine**: Algoritmos implementados en TypeScript puro (sin dependencias externas)
- **Deploy**: GitHub Pages (static export)

## 📄 Licencia

MIT — Úsalo libremente para educación.
