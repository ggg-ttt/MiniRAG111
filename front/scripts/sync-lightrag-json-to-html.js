const fs = require('fs');
const path = require('path');

const htmlPath = path.resolve(__dirname, '..', 'lightrag.html');
const jsonPath = path.resolve(__dirname, '..', 'lightrag.mock-data.json');

const html = fs.readFileSync(htmlPath, 'utf8');
const payload = JSON.parse(fs.readFileSync(jsonPath, 'utf8'));

function replaceBetweenMarkers(source, startMarker, endMarker, replacement) {
    const start = source.indexOf(startMarker);
    const end = source.indexOf(endMarker);
    if (start === -1 || end === -1 || end < start) {
        throw new Error(`Missing markers: ${startMarker} ... ${endMarker}`);
    }

    const contentStart = start + startMarker.length;
    return source.slice(0, contentStart) + replacement + source.slice(end);
}

const mockDataLiteral = ` ${JSON.stringify(payload.mockData || {}, null, 8)} `;
const domBindingsLiteral = ` ${JSON.stringify(payload.domBindings || [], null, 8)} `;

let nextHtml = html;
nextHtml = replaceBetweenMarkers(
    nextHtml,
    '/* LIGHTRAG_MOCK_DATA_START */',
    '/* LIGHTRAG_MOCK_DATA_END */',
    mockDataLiteral
);
nextHtml = replaceBetweenMarkers(
    nextHtml,
    '/* LIGHTRAG_DOM_BINDINGS_START */',
    '/* LIGHTRAG_DOM_BINDINGS_END */',
    domBindingsLiteral
);

fs.writeFileSync(htmlPath, nextHtml, 'utf8');
console.log(`Synced ${path.relative(process.cwd(), jsonPath)} -> ${path.relative(process.cwd(), htmlPath)}`);
