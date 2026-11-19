# Bibliotecas incluidas

`three.bundle.min.js` es [three.js](https://threejs.org) **r187** (licencia MIT, ver `LICENSE-three.txt`)
junto con `OrbitControls`, empaquetados en un único módulo ES para que la vista 3D funcione sin internet.

Para regenerarlo (requiere Node.js):

```bash
mkdir /tmp/three && cd /tmp/three
npm install three@0.187.0
printf "export * from 'three';\nexport { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';\n" > entry.js
npx esbuild entry.js --bundle --format=esm --minify --legal-comments=none --outfile=three.bundle.min.js
```
