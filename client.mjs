// client.mjs
import fetch from 'node-fetch';
import inquirer from 'inquirer';
import chalk from 'chalk';

const API_BASE = 'http://localhost:8000';

async function trainModel() {
  console.log(chalk.blue('Training AI Model...'));
  const res = await fetch(`${API_BASE}/train`, { method: 'POST' });
  const data = await res.json();
  console.log(chalk.green('Response:'), data.message);
}

async function adversarialTraining() {
  console.log(chalk.blue('Running Adversarial Training...'));
  const res = await fetch(`${API_BASE}/adversarial_training`, { method: 'POST' });
  const data = await res.json();
  console.log(chalk.green('Response:'), data.message);
}

async function threatAssessment() {
  console.log(chalk.blue('Analyzing AI Security Threats...'));
  const res = await fetch(`${API_BASE}/threat_assessment`);
  const data = await res.json();
  const riskLevel = data.risk_level || 'Unknown';
  const riskText = riskLevel === 'Low Risk' ? chalk.green(riskLevel) : chalk.red(riskLevel);
  console.log(chalk.yellow('Predicted Risk Level →'), riskText);
}

async function mainMenu() {
  const choices = [
    'Train AI Model',
    'Adversarial Training',
    'Threat Assessment',
    'Exit',
  ];

  while (true) {
    const answer = await inquirer.prompt([
      { type: 'list', name: 'action', message: 'Select an action:', choices }
    ]);

    switch (answer.action) {
      case 'Train AI Model':
        await trainModel();
        break;
      case 'Adversarial Training':
        await adversarialTraining();
        break;
      case 'Threat Assessment':
        await threatAssessment();
        break;
      case 'Exit':
        console.log(chalk.cyan('Goodbye!'));
        process.exit(0);
    }
  }
}

console.log(chalk.bold.green('\nWelcome to the AI Audit Platform CLI Frontend!\n'));
mainMenu();