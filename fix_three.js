const fs = require('fs');
let code = fs.readFileSync('src/render/sphereView.ts', 'utf-8');
code = code.replace(/import \{ OrbitControls \} from 'three\/examples\/jsm\/controls\/OrbitControls';/, "import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';");
code = code.replace(/THREE\.WebGPURenderer/, "THREE.WebGLRenderer");
code = code.replace(/THREE\.NodeMaterial/, "THREE.Material");
fs.writeFileSync('src/render/sphereView.ts', code);
