import fs from 'fs';

let content = fs.readFileSync('README.md', 'utf8');

const additionalContent = `
## 安装与依赖管理

项目核心可视化部分（\`index.html\` 和 \`src/\`）为纯前端实现，无须安装依赖即可直接在浏览器中运行。

但在进行本地开发（如运行测试用例）时，我们需要使用 Node.js 管理依赖：

\`\`\`bash
# 确保你安装了 Node.js (推荐 v18+)
# 安装开发依赖（主要是 Jest 测试框架）
npm install
\`\`\`

## 运行程序

为了解决浏览器的 CORS 跨域限制（用于加载 Web Worker），必须通过静态服务器运行，**不要直接双击打开 index.html**。

\`\`\`bash
# 使用 Python 启动服务
python -m http.server 8080

# 或者使用 npx
npx serve .
\`\`\`
随后在浏览器中打开 http://localhost:8080 即可体验。

## 运行测试

本项目使用了 \`Jest\` 进行单元测试（包含数学精度测试、流体力学物理特性测试和时间推进收敛性测试）。

由于项目使用原生 ES Modules，运行测试必须带上 \`--experimental-vm-modules\` 标志。你可以直接使用 npm 脚本：

\`\`\`bash
# 运行所有测试套件
npm run test
\`\`\`

测试用例位于 \`__tests__/\` 目录下，与 \`program.md\` 中的测试计划（UT-01~04, PT-01~04, CT-03）一一对应。
`;

// Insert the additional content before the "功能特性" section
content = content.replace('## 功能特性', additionalContent + '\n## 功能特性');

fs.writeFileSync('README.md', content);
console.log('README updated.');
