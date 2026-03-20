const fs = require('fs');
const path = require('path');
const vm = require('vm');

const htmlPath = path.resolve(__dirname, '..', 'lightrag.html');
const jsonPath = path.resolve(__dirname, '..', 'lightrag.mock-data.json');

const html = fs.readFileSync(htmlPath, 'utf8');

function mustMatch(pattern, source, label) {
    const match = source.match(pattern);
    if (!match) {
        throw new Error(`Failed to extract ${label}`);
    }
    return match;
}

function cleanText(value) {
    return value.replace(/<[^>]+>/g, '').replace(/\s+/g, ' ').trim();
}

function decodeJsString(value) {
    return vm.runInNewContext(`"${value.replace(/\\/g, '\\\\').replace(/"/g, '\\"')}"`);
}

function extractAttr(id, attr) {
    const match = mustMatch(
        new RegExp(`<[^>]*id="${id}"[^>]*\\s${attr}="([^"]*)"[^>]*>`, 's'),
        html,
        `${id}.${attr}`
    );
    return decodeJsString(match[1]);
}

function extractTextById(id) {
    const match = mustMatch(
        new RegExp(`<[^>]*id="${id}"[^>]*>([\\s\\S]*?)</[^>]+>`, 's'),
        html,
        `${id}.text`
    );
    return cleanText(match[1]);
}

function extractChatWelcome() {
    const match = mustMatch(
        /<div class="chat-container[\s\S]*?<p class="font-semibold mb-2">([\s\S]*?)<\/p>[\s\S]*?<p>([\s\S]*?)<\/p>[\s\S]*?<ul class="mt-2 space-y-1 text-xs">([\s\S]*?)<\/ul>[\s\S]*?<p class="mt-2 text-xs text-indigo-600">([\s\S]*?)<\/p>/,
        html,
        'chat welcome'
    );

    const bullets = Array.from(match[3].matchAll(/<li>([\s\S]*?)<\/li>/g)).map(item => cleanText(item[1]));
    return {
        title: cleanText(match[1]),
        description: cleanText(match[2]),
        bullets,
        footer: cleanText(match[4])
    };
}

function extractEmptyState() {
    const match = mustMatch(
        /<div id="entity-detail"[\s\S]*?<i class="([^"]+) text-4xl mb-4"><\/i>[\s\S]*?<p>([\s\S]*?)<\/p>/,
        html,
        'entity empty state'
    );
    return {
        icon: match[1].trim(),
        text: cleanText(match[2])
    };
}

function extractFallbackEntities() {
    const match = mustMatch(
        /entitiesToDisplay\s*=\s*(?:mockData\.entities\?\.items\s*\|\|\s*)?\[(.*?)\];/s,
        html,
        'fallback entities'
    );
    return vm.runInNewContext(`[${match[1]}]`);
}

function buildDomBindings() {
    return [
        { selector: '#token-count', property: 'textContent', value: '0' },
        { selector: '#retrieval-time', property: 'textContent', value: '0' },
        { selector: '#generation-time', property: 'textContent', value: '0' },
        { selector: '#monitor-tab .rounded-xl.border.border-emerald-100.bg-emerald-50 .flex.items-center.justify-between.text-sm span:last-child', property: 'textContent', value: '稳定' },
        { selector: '#monitor-tab .rounded-xl.border.border-emerald-100.bg-emerald-50 p', property: 'textContent', value: '检索、图谱构建与日志模块均处于在线状态。' },
        { selector: '#monitor-tab .rounded-xl.bg-slate-50:nth-of-type(2) .flex.items-center.justify-between.text-xs span:last-child', property: 'textContent', value: '38%' },
        { selector: '#monitor-tab .rounded-xl.bg-slate-50:nth-of-type(2) .mt-2.h-2.rounded-full.bg-slate-200 > div', property: 'style.width', value: '38%' },
        { selector: '#monitor-tab .rounded-xl.bg-slate-50:nth-of-type(3) .flex.items-center.justify-between.text-xs span:last-child', property: 'textContent', value: '72%' },
        { selector: '#monitor-tab .rounded-xl.bg-slate-50:nth-of-type(3) .mt-2.h-2.rounded-full.bg-slate-200 > div', property: 'style.width', value: '72%' },
        { selector: '#monitor-tab .rounded-xl.bg-slate-50:nth-of-type(4) .flex.items-center.justify-between.text-xs span:last-child', property: 'textContent', value: '54%' },
        { selector: '#monitor-tab .rounded-xl.bg-slate-50:nth-of-type(4) .mt-2.h-2.rounded-full.bg-slate-200 > div', property: 'style.width', value: '54%' }
    ];
}

const data = {
    mockData: {
        chat: {
            welcome: extractChatWelcome(),
            inputPlaceholder: extractAttr('user-input', 'placeholder')
        },
        monitor: {
            summary: {
                nodes: extractTextById('monitor-nodes'),
                edges: extractTextById('monitor-edges'),
                latency: extractTextById('monitor-latency'),
                p95: extractTextById('monitor-p95'),
                memory: extractTextById('monitor-memory'),
                gpu: extractTextById('monitor-gpu')
            },
            logs: []
        },
        entities: {
            emptyState: extractEmptyState(),
            items: extractFallbackEntities()
        },
        documents: {
            items: []
        },
        graph: {
            elements: []
        },
        graphImport: {
            formats: {}
        }
    },
    domBindings: buildDomBindings()
};

fs.writeFileSync(jsonPath, JSON.stringify(data, null, 2) + '\n', 'utf8');
console.log(`Wrote ${path.relative(process.cwd(), jsonPath)}`);
