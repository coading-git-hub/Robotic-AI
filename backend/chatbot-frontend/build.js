#!/usr/bin/env node
// Build script for the chatbot frontend

const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');

console.log('Building chatbot frontend...');

try {
  // Check if package.json exists
  const packageJsonPath = path.join(__dirname, 'package.json');
  if (!fs.existsSync(packageJsonPath)) {
    throw new Error('package.json not found. Please run this script from the project root.');
  }

  // Read package.json
  const packageJson = JSON.parse(fs.readFileSync(packageJsonPath, 'utf8'));

  console.log(`Building ${packageJson.name} v${packageJson.version}...`);

  // Check if node_modules exists, if not run npm install
  const nodeModulesPath = path.join(__dirname, 'node_modules');
  if (!fs.existsSync(nodeModulesPath)) {
    console.log('Installing dependencies...');
    execSync('npm install', { stdio: 'inherit', cwd: __dirname });
  }

  // Run the build command
  console.log('Running build...');
  execSync('npm run build', { stdio: 'inherit', cwd: __dirname });

  console.log('Build completed successfully!');
  console.log('The built files are in the build/ directory.');
} catch (error) {
  console.error('Build failed:', error.message);
  process.exit(1);
}