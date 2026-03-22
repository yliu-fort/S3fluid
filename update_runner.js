const fs = require('fs');
let code = fs.readFileSync('tests/playwright/runner.ts', 'utf-8');

const imports = "import filterSpectrumSrc from '../../src/shaders/filterSpectrum.wgsl?raw';\n" +
"import initRandomSrc from '../../src/shaders/initRandom.wgsl?raw';\n" +
"import velocityFromPsiSrc from '../../src/shaders/velocityFromPsi.wgsl?raw';\n" +
"import advectGridSrc from '../../src/shaders/advectGrid.wgsl?raw';\n" +
"import rhsComposeSrc from '../../src/shaders/rhsCompose.wgsl?raw';\n" +
"import rk4StageSrc from '../../src/shaders/rk4Stage.wgsl?raw';\n" +
"import rk4CombineSrc from '../../src/shaders/rk4Combine.wgsl?raw';\n" +
"import energyIntegrandSrc from '../../src/shaders/energyIntegrand.wgsl?raw';\n" +
"import reduceSumSrc from '../../src/shaders/reduceSum.wgsl?raw';\n";

code = code.replace(/import filterSpectrumSrc from '\.\.\/\.\.\/src\/shaders\/filterSpectrum\.wgsl\?raw';/, imports);

const inits = "await this.pipeline.init(\n" +
"            fftForwardLonSrc,\n" +
"            fftInverseLonSrc,\n" +
"            legendreAnalysisSrc,\n" +
"            legendreSynthesisSrc,\n" +
"            legendreSynthesisDThetaSrc,\n" +
"            mulIMSrc,\n" +
"            applyLaplacianSrc,\n" +
"            invertLaplacianSrc,\n" +
"            filterSpectrumSrc,\n" +
"            initRandomSrc,\n" +
"            velocityFromPsiSrc,\n" +
"            advectGridSrc,\n" +
"            rhsComposeSrc,\n" +
"            rk4StageSrc,\n" +
"            rk4CombineSrc,\n" +
"            energyIntegrandSrc,\n" +
"            reduceSumSrc\n" +
"        );";

code = code.replace(/await this\.pipeline\.init\([\s\S]*?filterSpectrumSrc\n\s*\);/, inits);

fs.writeFileSync('tests/playwright/runner.ts', code);
