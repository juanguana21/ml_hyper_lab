# 🚀 Guía de Deploy en GitHub Pages — ML HyperLab

Esta guía explica cómo publicar ML HyperLab en GitHub Pages para que tus estudiantes puedan acceder desde cualquier navegador sin instalar nada.

## Requisitos Previos

- Una cuenta de GitHub
- Git instalado en tu computadora
- Node.js 20+ instalado (solo para desarrollo local)

---

## Paso 1: Crear el repositorio en GitHub

1. Ve a [github.com/new](https://github.com/new)
2. Nombra el repositorio: **`ml-hyperlab`** (importante: el nombre del repo define la URL)
3. Marca como **Público** (requerido para GitHub Pages gratis)
4. **NO** inicialices con README, .gitignore o license
5. Click en **Create repository**

> ⚠️ El nombre del repositorio es importante porque define la URL final:
> - Si el repo es `tu-usuario/ml-hyperlab` → la URL será `https://tu-usuario.github.io/ml-hyperlab/`
> - Si usas otro nombre, la app se ajusta automáticamente gracias a la configuración de `basePath`

---

## Paso 2: Subir el código al repositorio

Desde la carpeta del proyecto, ejecuta:

```bash
# Inicializar git (si no está inicializado)
git init

# Agregar el remote de tu repositorio
git remote add origin https://github.com/TU-USUARIO/ml-hyperlab.git

# Agregar todos los archivos
git add .

# Hacer el primer commit
git commit -m "ML HyperLab - Laboratorio Interactivo de Machine Learning"

# Subir a GitHub
git branch -M main
git push -u origin main
```

---

## Paso 3: Activar GitHub Pages

1. Ve a tu repositorio en GitHub
2. Click en **Settings** (Configuración)
3. En el menú izquierdo, click en **Pages**
4. En **Source**, selecciona **GitHub Actions**
5. ¡Listo! No necesitas seleccionar ninguna rama, el workflow se encarga de todo

---

## Paso 4: Verificar el deploy automático

El archivo `.github/workflows/deploy.yml` ya está configurado para:

1. Detectar cada `push` a la rama `main`
2. Instalar dependencias con Bun
3. Construir el sitio estático (`next build`)
4. Agregar `.nojekyll` (para que GitHub Pages no rompa las carpetas que empiezan con `_`)
5. Desplegar automáticamente a GitHub Pages

Para verificar:

1. Ve a la pestaña **Actions** en tu repositorio
2. Deberías ver un workflow corriendo llamado "Deploy ML HyperLab to GitHub Pages"
3. Espera a que termine (verde ✅)
4. La URL será: `https://TU-USUARIO.github.io/ml-hyperlab/`

---

## Paso 5: Compartir con los estudiantes

¡Eso es todo! Comparte la URL con tus estudiantes:

```
https://TU-USUARIO.github.io/ml-hyperlab/
```

Los estudiantes solo necesitan:
- Un navegador moderno (Chrome, Firefox, Edge, Safari)
- Conexión a internet (solo para cargar la página la primera vez)
- **No necesitan instalar Python, Node.js, ni nada**

---

## Actualizar la aplicación

Cada vez que hagas cambios y subas al repositorio:

```bash
git add .
git commit -m "Descripción del cambio"
git push
```

El deploy automático se ejecutará y en ~2 minutos la página estará actualizada.

---

## Estructura de archivos clave

```
ml-hyperlab/
├── .github/workflows/deploy.yml   ← Workflow de deploy automático
├── public/.nojekyll               ← Evita que GitHub Pages ignore carpetas _
├── next.config.ts                 ← Configuración con basePath dinámico
├── package.json                   ← Script build:static
├── src/
│   ├── app/page.tsx               ← App principal (3 pestañas)
│   ├── lib/
│   │   ├── ml-algorithms.ts       ← Algoritmos ML en TypeScript puro
│   │   ├── ml-config.ts           ← Configuración de algoritmos y datasets
│   │   └── workshops.ts           ← Talleres guiados
│   └── components/ui/             ← Componentes shadcn/ui
```

---

## Solución de Problemas

### La página se ve en blanco o no carga

- Verifica que el nombre del repositorio coincida con el `basePath`
- El `next.config.ts` detecta automáticamente el nombre del repo desde `GITHUB_REPOSITORY`
- Si usas un dominio custom (ej: `ml-hyperlab.com`), necesitas quitar el basePath

### Error 404 en la URL

- Asegúrate de que GitHub Pages esté configurado con **Source: GitHub Actions** (no "Deploy from branch")
- Verifica que el workflow se completó exitosamente en la pestaña Actions

### Los estilos no cargan

- Verifica que el archivo `.nojekyll` exista en `public/`
- Sin `.nojekyll`, GitHub Pages ignora carpetas que empiezan con `_` (como `_next/`)

### Quiero usar un dominio custom

1. En Settings → Pages → Custom domain, agrega tu dominio
2. Modifica `next.config.ts` para dejar `basePath` vacío:

```typescript
const basePath = "";  // Dominio custom, sin basePath
```

3. Haz push y espera al redeploy

---

## Despliegue local (alternativa)

Si prefieres no usar GitHub Pages, también puedes servir los archivos estáticos localmente:

```bash
# Construir
bun run build:static

# Servir con cualquier servidor estático
npx serve out/
# o
python3 -m http.server 8000 --directory out/
```

---

## Notas técnicas

- **Todo corre en el navegador**: Los algoritmos de ML (KNN, Random Forest, XGBoost, etc.) están implementados en TypeScript puro. No hay backend.
- **Tamaño del build**: ~1.5MB (muy ligero)
- **Compatibilidad**: Funciona en cualquier navegador moderno, incluyendo móviles
- **Privacidad**: Los datos no salen del navegador del estudiante
