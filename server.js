#!/usr/bin/env node

const http = require('http');
const fs = require('fs');
const path = require('path');
const { spawn } = require('child_process');

const PORT = 3000;

function serveStatic(res, filePath, contentType) {
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
      serveStatic(res, path.join(__dirname, 'index.html'), 'text/html');
    } else if (req.url === '/style.css') {
      serveStatic(res, path.join(__dirname, 'style.css'), 'text/css');
    } else {
      res.writeHead(404);
      res.end('Not found');
    }
  }
  else if (req.method === 'POST' && req.url === '/ingest') {
    let body = '';
    req.on('data', chunk => { body += chunk.toString(); });
    req.on('end', () => {
      const params = new URLSearchParams(body);
      const chunkSize = params.get('chunkSize') || '128';
      const folderpath = params.get('folderpath') || '';
      const urlpath = params.get('urlpath') || '';
      const filetext = params.get('filetext') || '';

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

      const pyProcess = spawn('python', ['rag_inference.py', 'ingest']);
      pyProcess.stdin.write(JSON.stringify(ingestionPayload));
      pyProcess.stdin.end();

      let resultData = '';
      pyProcess.stdout.on('data', data => {
        resultData += data.toString();
      });

      pyProcess.stderr.on('data', data => {
        console.error('Python Ingest Error:', data.toString());
      });

      pyProcess.on('close', code => {
        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ status: "ingested", details: resultData }, null, 2));
      });
    });
  }
  else if (req.method === 'POST' && req.url === '/inference') {
    let body = '';
    req.on('data', chunk => { body += chunk.toString(); });
    req.on('end', () => {
      const params = new URLSearchParams(body);
      const userQuery = params.get('query') || 'No query';
      const backend = params.get('backend') || 'local';

      const pyProcess = spawn('python', ['rag_inference.py', userQuery, backend]);

      let resultData = '';
      pyProcess.stdout.on('data', data => {
        resultData += data.toString();
      });

      pyProcess.stderr.on('data', data => {
        console.error('Python Error:', data.toString());
      });

      pyProcess.on('close', code => {
        try {
          const jsonResponse = JSON.parse(resultData);
          res.writeHead(200, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify(jsonResponse, null, 2));
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
