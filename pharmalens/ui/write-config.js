const fs = require("fs");
const path = require("path");

const apiUrl = process.env.PHARMALENS_API_URL || "";
const output = `window.PHARMALENS_API_URL = ${JSON.stringify(apiUrl)};\n`;

fs.writeFileSync(path.join(__dirname, "config.js"), output);
console.log(`Wrote config.js with PHARMALENS_API_URL=${apiUrl || "(same-origin)"}`);
