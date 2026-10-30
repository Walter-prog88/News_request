# 依赖说明

运行环境：支持 HTML5、SVG、Canvas、ES6 JavaScript 的现代浏览器。无构建步骤，无 npm 安装步骤。

`lessons/01-gridworld.html` 至 `lessons/05-stale-data.html` 引用以下固定版本资源：

- `https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.9/katex.min.css`
- `https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.9/katex.min.js`
- `https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.9/contrib/auto-render.min.js`

CSS 还会加载 KaTeX 对应字体。CDN 不可用时，公式保留为原始 LaTeX 文本，正文仍可阅读。若要离线排版，可按 KaTeX 官方安装说明下载完整发行包与字体，将上述引用改为本地相对路径，并保留其 LICENSE。

其余页面、目录入口均自包含，无第三方 JavaScript 库。项目没有网络采集、模型调用、账户登录或服务端代码。

内容检查脚本只使用 Python 3 标准库。JavaScript 语法检查另需 Node.js；Node.js 不参与页面运行。
