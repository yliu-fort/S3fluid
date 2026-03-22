const { execSync } = require('child_process');
try {
  const result = execSync('xvfb-run npx playwright test tests/playwright/ --reporter=list');
  console.log(result.toString());
} catch(e) {
  console.log(e.stdout.toString());
  console.log(e.stderr.toString());
}
