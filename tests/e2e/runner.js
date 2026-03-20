import { chromium } from 'playwright';
import * as http from 'http';
import * as fs from 'fs';
import * as path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const rootDir = path.resolve(__dirname, '../../');

// Create a static HTTP server to serve the repo files
const server = http.createServer((req, res) => {
    // Strip query strings if any (e.g., from module imports)
    let reqPath = req.url.split('?')[0];
    let filePath = path.join(rootDir, reqPath === '/' ? 'tests/e2e/gpu_physics.html' : reqPath);

    // Quick and dirty mime-type resolution
    let extname = path.extname(filePath);
    let contentType = 'text/html';
    switch (extname) {
        case '.js': contentType = 'text/javascript'; break;
        case '.css': contentType = 'text/css'; break;
        case '.json': contentType = 'application/json'; break;
        case '.png': contentType = 'image/png'; break;
        case '.jpg': contentType = 'image/jpg'; break;
    }

    fs.readFile(filePath, (error, content) => {
        if (error) {
            if(error.code == 'ENOENT'){
                res.writeHead(404);
                res.end('File not found', 'utf-8');
            } else {
                res.writeHead(500);
                res.end('Server error: '+error.code, 'utf-8');
            }
        } else {
            res.writeHead(200, { 'Content-Type': contentType });
            res.end(content, 'utf-8');
        }
    });
});

const PORT = 8000;

async function runTests() {
    server.listen(PORT, '127.0.0.1');

    console.log(`Running static server on http://127.0.0.1:${PORT}`);

    // Launch Playwright browser
    const browser = await chromium.launch({
        args: [
            '--no-sandbox',
            '--disable-setuid-sandbox',
            '--use-gl=angle', // Use software OpenGL backend inside headless
            '--use-angle=swiftshader'
        ]
    });

    const page = await browser.newPage();

    // Forward console logs to terminal
    page.on('console', msg => console.log(`[Browser] ${msg.text()}`));
    page.on('pageerror', err => console.error(`[Browser Error] ${err}`));

    await page.goto(`http://127.0.0.1:${PORT}`);

    console.log('Waiting for tests to finish...');

    // Wait until the status div has 'data-completed' set to "true"
    await page.waitForFunction(() => {
        const el = document.getElementById('status');
        return el && el.dataset.completed === "true";
    }, { timeout: 30000 });

    const failedCount = await page.evaluate(() => {
        const el = document.getElementById('status');
        return parseInt(el.dataset.failed || '0', 10);
    });

    console.log('Failed Tests:', failedCount);

    await browser.close();
    server.close();

    if (failedCount > 0) {
        console.error('❌ E2E GPU Physics Tests Failed!');
        process.exit(1);
    } else {
        console.log('✅ All E2E GPU Physics Tests Passed!');
        process.exit(0);
    }
}

runTests().catch(e => {
    console.error('Runner Error:', e);
    process.exit(1);
});
