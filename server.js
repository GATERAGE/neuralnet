#!/usr/bin/env node

/**
 * Node.js server that hosts the RAGE pipeline endpoints (/ingest, /inference),
 * logs “Python Ingest: Batches…” for batch progress lines from Python
 * rather than labeling them as errors.
 */

const http = require('http');
const fs = require('fs');
const { spawn } = require('child_process');
const path = require('path');

const PORT = 3000;

// Use an environment variable to specify Python path, fallback to 'python'
const pythonExecutable = process.env.PYTHON_PATH || 'python';
console.log('Using Python executable:', pythonExecutable);

const memoryFolder = path.join(__dirname, 'memory');
if (!fs.existsSync(memoryFolder)) {
  fs.mkdirSync(memoryFolder);
}

/**
 * Serves static files (index.html, style.css) for minimal front-end testing
 */
function serveFile(res, filePath, contentType) {
  fs.readFile(filePath, (err, data) => {
    if (err) {
      res.writeHead(500);
      return res.end(`Error loading ${filePath}`);
    }
    res.writeHead(200, { 'Content-Type': contentType });
    res.end(data);
  });
}

const server = http.createServer((req, res) => {
  if (req.method === 'GET') {
    if (req.url === '/' || req.url === '/index.html') {
      serveFile(res, path.join(__dirname, 'index.html'), 'text/html');
    } else if (req.url === '/style.css') {
      serveFile(res, path.join(__dirname, 'style.css'), 'text/css');
    } else {
      res.writeHead(404);
      res.end('Not found');
    }
  }
  else if (req.method === 'POST' && req.url === '/ingest') {
    // /ingest route for data ingestion
    let body = '';
    req.on('data', chunk => { body += chunk.toString(); });
    req.on('end', () => {
      const params = new URLSearchParams(body);

      const folderpath = params.get('folderpath') || '';
      const urlpath = params.get('urlpath') || '';
      const filetext = params.get('filetext') || '';
      const chunkSize = params.get('chunkSize') || '128';

      // Possibly save user-pasted text to docs/ folder
      let tempFilePath = null;
      if (filetext.trim().length > 0) {
        const docsFolder = path.join(__dirname, 'docs');
        if (!fs.existsSync(docsFolder)) fs.mkdirSync(docsFolder);
        tempFilePath = path.join(docsFolder, `pasted_${Date.now()}.txt`);
        fs.writeFileSync(tempFilePath, filetext, 'utf-8');
      }

      const ingestionPayload = {
        chunk_size: parseInt(chunkSize, 10),
        filepaths: tempFilePath ? [tempFilePath] : [],
        folderpaths: folderpath ? [folderpath] : [],
        urls: urlpath ? [urlpath] : []
      };

      // Spawn Python ingestion script
      const pyProcess = spawn(pythonExecutable, ['rag_inference.py', 'ingest']);
      pyProcess.stdin.write(JSON.stringify(ingestionPayload));
      pyProcess.stdin.end();

      let resultData = '';
      pyProcess.stdout.on('data', (data) => {
        resultData += data.toString();
      });

      pyProcess.stderr.on('data', (data) => {
        const stderr = data.toString().trim();
        if (stderr.includes('Batches:')) {
          // Show as progress line
          console.log(`Python Ingest: ${stderr}`);
        } else if (stderr) {
          // Actual error line
          console.error(`Python Ingest Error: ${stderr}`);
        }
      });

      pyProcess.on('close', (code) => {
        const memFileName = path.join(memoryFolder, `ingest_${Date.now()}.json`);
        fs.writeFileSync(memFileName, JSON.stringify({ ingestionPayload, resultData }, null, 2));

        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ status: "ingested", memoryFile: memFileName, details: resultData }, null, 2));
      });
    });
  }
  else if (req.method === 'POST' && req.url === '/inference') {
    // /inference route for final query
    let body = '';
    req.on('data', chunk => { body += chunk.toString(); });
    req.on('end', () => {
      const params = new URLSearchParams(body);
      const userQuery = params.get('query') || 'No query provided';
      const backend = params.get('backend') || 'local';

      const pyProcess = spawn(pythonExecutable, ['rag_inference.py', userQuery, backend]);

      let resultData = '';
      pyProcess.stdout.on('data', (data) => {
        resultData += data.toString();
      });

      pyProcess.stderr.on('data', (data) => {
        const stderr = data.toString().trim();
        // For inference, typically no "Batches:" lines, but let's handle similarly
        if (stderr.includes('Batches:')) {
          console.log(`Python Ingest: ${stderr}`);
        } else if (stderr) {
          console.error(`Python Error: ${stderr}`);
        }
      });

      pyProcess.on('close', (code) => {
        try {
          const jsonResponse = JSON.parse(resultData);
          const memFileName = path.join(memoryFolder, `inference_${Date.now()}.json`);
          fs.writeFileSync(memFileName, JSON.stringify(jsonResponse, null, 2));

          res.writeHead(200, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify({ memoryFile: memFileName, ...jsonResponse }, null, 2));
        } catch (err) {
          console.error('Error parsing Python output:', err);
          res.writeHead(500);
          res.end('Internal Server Error');
        }
      });
    });
  }
  else {
    res.writeHead(404);
    res.end('Not Found');
  }
});

server.listen(PORT, () => {
  console.log(`Server running at http://localhost:${PORT}/`);
});

