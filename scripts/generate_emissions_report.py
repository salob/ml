#!/usr/bin/env python3
"""
Generate HTML emissions report from readable emissions CSV

Usage: python scripts/generate_emissions_report.py
"""

import pandas as pd
from pathlib import Path

def generate_html_report():
    # Read the readable emissions CSV (CodeCarbon)
    codecarbon_path = Path('logs') / 'emissions_readable.csv'
    carbontracker_path = Path('logs') / 'carbontracker_readable.csv'
    output_path = Path('reports') / 'emissions_report.html'
    
    if not codecarbon_path.exists():
        print(f"Error: {codecarbon_path} not found. Run 'python scripts/convert_emissions.py' first.")
        return
    
    df_codecarbon = pd.read_csv(codecarbon_path)
    
    # Try to load CarbonTracker data (optional)
    df_carbontracker = None
    if carbontracker_path.exists():
        df_carbontracker = pd.read_csv(carbontracker_path)
        print(f"Found CarbonTracker data: {len(df_carbontracker)} runs")
    else:
        print(f"Note: {carbontracker_path} not found. Only CodeCarbon data will be shown.")
    
    # Load metrics CSVs
    logreg_metrics_path = Path('logs') / 'imdb_logreg_metrics.csv'
    cnn_metrics_path = Path('logs') / 'imdb_cnn_metrics.csv'
    transformer_metrics_path = Path('logs') / 'imdb_transformer_metrics.csv'
    
    df_logreg_metrics = None
    df_cnn_metrics = None
    df_transformer_metrics = None
    
    if logreg_metrics_path.exists():
        df_logreg_metrics = pd.read_csv(logreg_metrics_path)
        print(f"Found LogReg metrics: {len(df_logreg_metrics)} runs")
    
    if cnn_metrics_path.exists():
        df_cnn_metrics = pd.read_csv(cnn_metrics_path)
        print(f"Found CNN metrics: {len(df_cnn_metrics)} runs")
    
    if transformer_metrics_path.exists():
        df_transformer_metrics = pd.read_csv(transformer_metrics_path)
        print(f"Found Transformer metrics: {len(df_transformer_metrics)} runs")
    
    # Convert DataFrames to CSV strings for embedding
    codecarbon_csv = df_codecarbon.to_csv(index=False)
    carbontracker_csv = df_carbontracker.to_csv(index=False) if df_carbontracker is not None else ""
    logreg_metrics_csv = df_logreg_metrics.to_csv(index=False) if df_logreg_metrics is not None else ""
    cnn_metrics_csv = df_cnn_metrics.to_csv(index=False) if df_cnn_metrics is not None else ""
    transformer_metrics_csv = df_transformer_metrics.to_csv(index=False) if df_transformer_metrics is not None else ""
    
    # HTML template with embedded CSV
    html_template = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ML Model Emissions Report</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/date-fns@2.29.3/index.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chartjs-adapter-date-fns@3.0.0/dist/chartjs-adapter-date-fns.bundle.min.js"></script>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            min-height: 100vh;
        }

        .container {
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
            overflow: hidden;
        }

        header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }

        h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
            font-weight: 700;
        }

        .subtitle {
            font-size: 1.2em;
            opacity: 0.9;
        }

        .content {
            padding: 40px;
        }

        .summary-cards {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 40px;
        }

        .card {
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            padding: 25px;
            border-radius: 15px;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
            transition: transform 0.3s ease;
        }

        .card:hover {
            transform: translateY(-5px);
        }

        .card-title {
            font-size: 0.9em;
            color: #666;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 10px;
        }

        .card-value {
            font-size: 2em;
            font-weight: bold;
            color: #333;
        }

        .card-unit {
            font-size: 0.8em;
            color: #888;
            margin-left: 5px;
        }

        .chart-container {
            background: white;
            padding: 30px;
            border-radius: 15px;
            margin-bottom: 30px;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05);
        }

        .chart-title {
            font-size: 1.3em;
            font-weight: 600;
            margin-bottom: 20px;
            color: #333;
        }

        .chart-wrapper {
            position: relative;
            height: 400px;
        }

        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
            gap: 30px;
            margin-bottom: 30px;
        }

        .system-info {
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            color: white;
            padding: 30px;
            border-radius: 15px;
            margin-bottom: 30px;
        }

        .system-info h2 {
            margin-bottom: 20px;
        }

        .info-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
        }

        .info-item {
            background: rgba(255, 255, 255, 0.1);
            padding: 15px;
            border-radius: 10px;
        }

        .info-label {
            font-size: 0.85em;
            opacity: 0.8;
            margin-bottom: 5px;
        }

        .info-value {
            font-size: 1.1em;
            font-weight: 600;
        }

        footer {
            text-align: center;
            padding: 20px;
            color: #666;
            font-size: 0.9em;
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🌱 ML Model Emissions Report</h1>
            <p class="subtitle">Energy Consumption & Carbon Footprint Analysis</p>
        </header>

        <div class="content">
            <!-- Summary Cards -->
            <div class="summary-cards">
                <div class="card">
                    <div class="card-title">Total Experiments</div>
                    <div class="card-value" id="totalExperiments">0</div>
                </div>
                <div class="card">
                    <div class="card-title">Total Energy</div>
                    <div class="card-value" id="totalEnergy">0</div>
                </div>
                <div class="card">
                    <div class="card-title">Total CO₂ Emissions</div>
                    <div class="card-value" id="totalEmissions">0</div>
                </div>
                <div class="card">
                    <div class="card-title">Total Duration</div>
                    <div class="card-value" id="totalDuration">0</div>
                </div>
            </div>

            <!-- Charts Grid -->
            <div class="grid">
                <div class="chart-container">
                    <div class="chart-title">Average Energy Consumption by Model</div>
                    <div class="chart-wrapper">
                        <canvas id="energyChart"></canvas>
                    </div>
                </div>

                <div class="chart-container">
                    <div class="chart-title">Average CO₂ Emissions by Model</div>
                    <div class="chart-wrapper">
                        <canvas id="emissionsChart"></canvas>
                    </div>
                </div>
            </div>

            <div class="grid">
                <div class="chart-container">
                    <div class="chart-title">Average Training Duration by Model</div>
                    <div class="chart-wrapper">
                        <canvas id="durationChart"></canvas>
                    </div>
                </div>

                <div class="chart-container">
                    <div class="chart-title">Average Power Consumption Breakdown</div>
                    <div class="chart-wrapper">
                        <canvas id="powerChart"></canvas>
                    </div>
                </div>
            </div>

            <div class="grid">
                <div class="chart-container">
                    <div class="chart-title">Logistic Regression - Emissions per Run</div>
                    <div class="chart-wrapper">
                        <canvas id="logregEmissionsChart"></canvas>
                    </div>
                </div>

                <div class="chart-container">
                    <div class="chart-title">CNN - Emissions per Run</div>
                    <div class="chart-wrapper">
                        <canvas id="cnnEmissionsChart"></canvas>
                    </div>
                </div>
        

                <div class="chart-container">
                    <div class="chart-title">Transformer - Emissions per Run</div>
                    <div class="chart-wrapper">
                        <canvas id="transformerEmissionsChart"></canvas>
                    </div>
                </div>
            </div>

            <!-- Performance Metrics Section -->
            <h2 style="margin-top: 60px; margin-bottom: 30px; color: #333; font-size: 2em;">Model Performance Metrics</h2>
            
            <!-- Individual Model Performance -->
            <div class="grid">
                <div class="chart-container">
                    <div class="chart-title">Logistic Regression - Test Accuracy & F1 per Run</div>
                    <div class="chart-wrapper">
                        <canvas id="logregPerformanceChart"></canvas>
                    </div>
                </div>

                <div class="chart-container">
                    <div class="chart-title">CNN - Test Accuracy & F1 per Run</div>
                    <div class="chart-wrapper">
                        <canvas id="cnnPerformanceChart"></canvas>
                    </div>
                </div>

                <div class="chart-container">
                    <div class="chart-title">Transformer - Test Accuracy & F1 per Run</div>
                    <div class="chart-wrapper">
                        <canvas id="transformerPerformanceChart"></canvas>
                    </div>
                </div>
            </div>

            <!-- Cross-Model Comparison -->
            <h2 style="margin-top: 60px; margin-bottom: 30px; color: #333; font-size: 2em;">Cross-Model Comparison</h2>
            
            <div class="grid">
                <div class="chart-container">
                    <div class="chart-title">Average Test Accuracy Comparison</div>
                    <div class="chart-wrapper">
                        <canvas id="accuracyComparisonChart"></canvas>
                    </div>
                </div>

                <div class="chart-container">
                    <div class="chart-title">Average F1 Score Comparison</div>
                    <div class="chart-wrapper">
                        <canvas id="f1ComparisonChart"></canvas>
                    </div>
                </div>
            </div>

            <!-- Emissions vs Performance Correlation -->
            <h2 style="margin-top: 60px; margin-bottom: 30px; color: #333; font-size: 2em;">Emissions vs Performance Analysis</h2>
            
            <div class="grid">
                <div class="chart-container">
                    <div class="chart-title">Emissions vs Test Accuracy</div>
                    <div class="chart-wrapper">
                        <canvas id="emissionsVsAccuracyChart"></canvas>
                    </div>
                </div>

                <div class="chart-container">
                    <div class="chart-title">Emissions vs F1 Score</div>
                    <div class="chart-wrapper">
                        <canvas id="emissionsVsF1Chart"></canvas>
                    </div>
                </div>

                <div class="chart-container">
                    <div class="chart-title">Energy Efficiency (Accuracy per Watt-hour)</div>
                    <div class="chart-wrapper">
                        <canvas id="efficiencyChart"></canvas>
                    </div>
                </div>
            </div>

            <!-- System Info -->
            <div class="system-info">
                <h2>System Information</h2>
                <div class="info-grid">
                    <div class="info-item">
                        <div class="info-label">CPU</div>
                        <div class="info-value" id="cpuModel">-</div>
                    </div>
                    <div class="info-item">
                        <div class="info-label">GPU</div>
                        <div class="info-value" id="gpuModel">-</div>
                    </div>
                    <div class="info-item">
                        <div class="info-label">CPU Cores</div>
                        <div class="info-value" id="cpuCount">-</div>
                    </div>
                    <div class="info-item">
                        <div class="info-label">RAM</div>
                        <div class="info-value" id="ramSize">-</div>
                    </div>
                    <div class="info-item">
                        <div class="info-label">Location</div>
                        <div class="info-value" id="location">-</div>
                    </div>
                    <div class="info-item">
                        <div class="info-label">Python Version</div>
                        <div class="info-value" id="pythonVersion">-</div>
                    </div>
                </div>
            </div>
        </div>

        <footer>
            <p>Generated on <span id="reportDate"></span> | Powered by CodeCarbon</p>
        </footer>
    </div>

    <script>
        // CSV Data embedded directly
        const codecarbonCSV = `CODECARBON_CSV_PLACEHOLDER`;
        const carbontrackerCSV = `CARBONTRACKER_CSV_PLACEHOLDER`;
        const logregMetricsCSV = `LOGREG_METRICS_CSV_PLACEHOLDER`;
        const cnnMetricsCSV = `CNN_METRICS_CSV_PLACEHOLDER`;
        const transformerMetricsCSV = `TRANSFORMER_METRICS_CSV_PLACEHOLDER`;

        // Parse CSV
        function parseCSV(csv) {
            if (!csv || csv.trim() === '') return [];
            const lines = csv.trim().split('\\n');
            const headers = lines[0].split(',');
            const data = [];
            
            for (let i = 1; i < lines.length; i++) {
                const obj = {};
                const currentLine = lines[i].split(',');
                
                for (let j = 0; j < headers.length; j++) {
                    obj[headers[j]] = currentLine[j];
                }
                data.push(obj);
            }
            
            return data;
        }

        const codecarbonData = parseCSV(codecarbonCSV);
        const carbontrackerData = parseCSV(carbontrackerCSV);
        
        console.log('CodeCarbon data length:', codecarbonData.length);
        console.log('CarbonTracker data length:', carbontrackerData.length);
        console.log('CarbonTracker sample:', carbontrackerData[0]);
        
        // Use CodeCarbon as primary data source for overview
        const data = codecarbonData;

        // Process data by model - flexible to handle both project_name and model_name
        function groupByModel(data) {
            const groups = {};
            data.forEach(row => {
                const model = row.project_name || row.model_name;  // Support both column names
                if (!groups[model]) {
                    groups[model] = [];
                }
                groups[model].push(row);
            });
            return groups;
        }

        const modelGroups = groupByModel(data);

        // Calculate statistics
        function calculateStats(data) {
            const stats = {
                totalExperiments: data.length,
                totalEnergy: 0,
                totalEmissions: 0,
                totalDuration: 0
            };

            data.forEach(row => {
                if (row.energy_consumed_wh && row.energy_consumed_wh.trim() !== '') {
                    stats.totalEnergy += parseFloat(row.energy_consumed_wh);
                }
                if (row.emissions_g && row.emissions_g.trim() !== '') {
                    stats.totalEmissions += parseFloat(row.emissions_g);
                }
                stats.totalDuration += parseFloat(row.duration) || 0;
            });

            return stats;
        }

        const overallStats = calculateStats(data);

        // Update summary cards
        document.getElementById('totalExperiments').textContent = overallStats.totalExperiments;
        document.getElementById('totalEnergy').innerHTML = overallStats.totalEnergy.toFixed(2) + '<span class="card-unit">Wh</span>';
        document.getElementById('totalEmissions').innerHTML = overallStats.totalEmissions.toFixed(3) + '<span class="card-unit">g</span>';
        document.getElementById('totalDuration').innerHTML = (overallStats.totalDuration / 3600).toFixed(2) + '<span class="card-unit">hrs</span>';

        // Update system info
        const firstRow = data[0];
        document.getElementById('cpuModel').textContent = firstRow.cpu_model;
        document.getElementById('gpuModel').textContent = firstRow.gpu_model;
        document.getElementById('cpuCount').textContent = firstRow.cpu_count + ' cores';
        document.getElementById('ramSize').textContent = firstRow.ram_total_size + ' GB';
        document.getElementById('location').textContent = firstRow.country_name + ', ' + firstRow.region;
        document.getElementById('pythonVersion').textContent = firstRow.python_version;
        document.getElementById('reportDate').textContent = new Date().toLocaleDateString('en-US', { year: 'numeric', month: 'long', day: 'numeric' });

        // Chart.js default configuration
        Chart.defaults.font.family = '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif';
        Chart.defaults.color = '#666';

        // Model colors
        const modelColors = {
            'IMDB_Transformer': 'rgba(102, 126, 234, 0.8)',
            'IMDB_LogReg': 'rgba(118, 75, 162, 0.8)',
            'IMDB_CNN': 'rgba(237, 100, 166, 0.8)'
        };

        const modelBorderColors = {
            'IMDB_Transformer': 'rgba(102, 126, 234, 1)',
            'IMDB_LogReg': 'rgba(118, 75, 162, 1)',
            'IMDB_CNN': 'rgba(237, 100, 166, 1)'
        };

        // Energy by Model Chart - use AVERAGES over runs
        const modelNames = Object.keys(modelGroups);
        const energyByModel = modelNames.map(model => {
            const rows = modelGroups[model].filter(row => row.energy_consumed_wh && row.energy_consumed_wh.trim() !== '');
            const total = rows.reduce((sum, row) => sum + parseFloat(row.energy_consumed_wh), 0);
            return rows.length > 0 ? total / rows.length : 0;
        });

        new Chart(document.getElementById('energyChart'), {
            type: 'bar',
            data: {
                labels: modelNames.map(m => m.replace('IMDB_', '')),
                datasets: [{
                    label: 'Energy Consumed (Wh)',
                    data: energyByModel,
                    backgroundColor: modelNames.map(m => modelColors[m]),
                    borderColor: modelNames.map(m => modelBorderColors[m]),
                    borderWidth: 2
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        callbacks: {
                            label: (context) => `Avg Energy: ${context.parsed.y.toFixed(4)} Wh`
                        }
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        title: { display: true, text: 'Average Energy (Wh)' }
                    }
                }
            }
        });

        // Emissions by Model Chart - use AVERAGES over runs
        const emissionsByModel = modelNames.map(model => {
            const rows = modelGroups[model].filter(row => row.emissions_g && row.emissions_g.trim() !== '');
            const total = rows.reduce((sum, row) => sum + parseFloat(row.emissions_g), 0);
            return rows.length > 0 ? total / rows.length : 0;
        });

        new Chart(document.getElementById('emissionsChart'), {
            type: 'bar',
            data: {
                labels: modelNames.map(m => m.replace('IMDB_', '')),
                datasets: [{
                    label: 'CO₂ Emissions (g)',
                    data: emissionsByModel,
                    backgroundColor: modelNames.map(m => modelColors[m]),
                    borderColor: modelNames.map(m => modelBorderColors[m]),
                    borderWidth: 2
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        callbacks: {
                            label: (context) => `Avg CO₂: ${context.parsed.y.toFixed(4)} g`
                        }
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        title: { display: true, text: 'Average CO₂ Emissions (g)' }
                    }
                }
            }
        });

        // Duration by Model Chart - use AVERAGES over runs
        const durationByModel = modelNames.map(model => {
            const rows = modelGroups[model];
            const total = rows.reduce((sum, row) => sum + (parseFloat(row.duration) || 0), 0);
            return rows.length > 0 ? (total / rows.length) / 60 : 0;
        });

        new Chart(document.getElementById('durationChart'), {
            type: 'bar',
            data: {
                labels: modelNames.map(m => m.replace('IMDB_', '')),
                datasets: [{
                    label: 'Training Duration (min)',
                    data: durationByModel,
                    backgroundColor: modelNames.map(m => modelColors[m]),
                    borderColor: modelNames.map(m => modelBorderColors[m]),
                    borderWidth: 2
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        callbacks: {
                            label: (context) => `Avg Duration: ${context.parsed.y.toFixed(2)} min`
                        }
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        title: { display: true, text: 'Average Duration (minutes)' }
                    }
                }
            }
        });

        // Power Consumption Chart
        const avgPowerByModel = modelNames.map(model => {
            const rows = modelGroups[model];
            const avgCpu = rows.reduce((sum, r) => sum + (parseFloat(r.cpu_power) || 0), 0) / rows.length;
            const avgGpu = rows.reduce((sum, r) => sum + (parseFloat(r.gpu_power) || 0), 0) / rows.length;
            const avgRam = rows.reduce((sum, r) => sum + (parseFloat(r.ram_power) || 0), 0) / rows.length;
            return { cpu: avgCpu, gpu: avgGpu, ram: avgRam };
        });

        new Chart(document.getElementById('powerChart'), {
            type: 'bar',
            data: {
                labels: modelNames.map(m => m.replace('IMDB_', '')),
                datasets: [
                    {
                        label: 'CPU Power',
                        data: avgPowerByModel.map(p => p.cpu),
                        backgroundColor: 'rgba(255, 99, 132, 0.7)',
                        borderColor: 'rgba(255, 99, 132, 1)',
                        borderWidth: 2
                    },
                    {
                        label: 'GPU Power',
                        data: avgPowerByModel.map(p => p.gpu),
                        backgroundColor: 'rgba(54, 162, 235, 0.7)',
                        borderColor: 'rgba(54, 162, 235, 1)',
                        borderWidth: 2
                    },
                    {
                        label: 'RAM Power',
                        data: avgPowerByModel.map(p => p.ram),
                        backgroundColor: 'rgba(75, 192, 192, 0.7)',
                        borderColor: 'rgba(75, 192, 192, 1)',
                        borderWidth: 2
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    tooltip: {
                        callbacks: {
                            label: (context) => `${context.dataset.label}: ${context.parsed.y.toFixed(4)} W`
                        }
                    }
                },
                scales: {
                    x: { stacked: false },
                    y: {
                        stacked: false,
                        beginAtZero: true,
                        title: { display: true, text: 'Average Power (W)' }
                    }
                }
            }
        });

        // Emissions per Run Charts - one for each model
        const logregRuns = modelGroups['IMDB_LogReg'] || [];
        const cnnRuns = modelGroups['IMDB_CNN'] || [];
        const transformerRuns = modelGroups['IMDB_Transformer'] || [];
        
        // Group CarbonTracker data by model
        const ctModelGroups = groupByModel(carbontrackerData);
        const ctLogregRuns = ctModelGroups['IMDB_Logreg'] || [];
        const ctCnnRuns = ctModelGroups['IMDB_Cnn'] || [];
        const ctTransformerRuns = ctModelGroups['IMDB_Transformer'] || [];
        
        console.log('CT Model Groups:', Object.keys(ctModelGroups));
        console.log('CT LogReg runs:', ctLogregRuns.length);
        console.log('CT CNN runs:', ctCnnRuns.length);
        console.log('CT Transformer runs:', ctTransformerRuns.length);

        // LogReg Emissions per Run
        if (logregRuns.length > 0) {
            const codecarbonEmissions = logregRuns.map(r => parseFloat(r.emissions_g) || 0);
            const carbontrackerEmissions = ctLogregRuns.map(r => parseFloat(r.co2_g_adjusted) || 0);
            
            console.log('LogReg - CodeCarbon emissions:', codecarbonEmissions);
            console.log('LogReg - CarbonTracker emissions:', carbontrackerEmissions);
            
            const datasets = [
                {
                    label: 'CodeCarbon',
                    data: codecarbonEmissions,
                    backgroundColor: 'rgba(118, 75, 162, 0.2)',
                    borderColor: 'rgba(118, 75, 162, 1)',
                    borderWidth: 3,
                    fill: true,
                    tension: 0.3,
                    pointRadius: 5,
                    pointHoverRadius: 7,
                    pointBackgroundColor: 'rgba(118, 75, 162, 1)',
                    pointBorderColor: '#fff',
                    pointBorderWidth: 2
                }
            ];
            
            if (carbontrackerEmissions.length > 0) {
                datasets.push({
                    label: 'CarbonTracker (PUE-adjusted)',
                    data: carbontrackerEmissions,
                    backgroundColor: 'rgba(255, 152, 0, 0.2)',
                    borderColor: 'rgba(255, 152, 0, 1)',
                    borderWidth: 3,
                    fill: true,
                    tension: 0.3,
                    pointRadius: 5,
                    pointHoverRadius: 7,
                    pointBackgroundColor: 'rgba(255, 152, 0, 1)',
                    pointBorderColor: '#fff',
                    pointBorderWidth: 2
                });
            }
            
            new Chart(document.getElementById('logregEmissionsChart'), {
                type: 'line',
                data: {
                    labels: logregRuns.map((_, i) => `Run ${i + 1}`),
                    datasets: datasets
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        tooltip: {
                            callbacks: {
                                label: (context) => `${context.dataset.label}: ${context.parsed.y.toFixed(6)} g`
                            }
                        }
                    },
                    scales: {
                        y: {
                            beginAtZero: true,
                            title: { display: true, text: 'CO₂ Emissions (g)' }
                        }
                    }
                }
            });
        }

        // CNN Emissions per Run
        if (cnnRuns.length > 0) {
            const codecarbonEmissions = cnnRuns.map(r => parseFloat(r.emissions_g) || 0);
            const carbontrackerEmissions = ctCnnRuns.map(r => parseFloat(r.co2_g_adjusted) || 0);
            
            const datasets = [
                {
                    label: 'CodeCarbon',
                    data: codecarbonEmissions,
                    backgroundColor: 'rgba(237, 100, 166, 0.2)',
                    borderColor: 'rgba(237, 100, 166, 1)',
                    borderWidth: 3,
                    fill: true,
                    tension: 0.3,
                    pointRadius: 5,
                    pointHoverRadius: 7,
                    pointBackgroundColor: 'rgba(237, 100, 166, 1)',
                    pointBorderColor: '#fff',
                    pointBorderWidth: 2
                }
            ];
            
            if (carbontrackerEmissions.length > 0) {
                datasets.push({
                    label: 'CarbonTracker (PUE-adjusted)',
                    data: carbontrackerEmissions,
                    backgroundColor: 'rgba(255, 152, 0, 0.2)',
                    borderColor: 'rgba(255, 152, 0, 1)',
                    borderWidth: 3,
                    fill: true,
                    tension: 0.3,
                    pointRadius: 5,
                    pointHoverRadius: 7,
                    pointBackgroundColor: 'rgba(255, 152, 0, 1)',
                    pointBorderColor: '#fff',
                    pointBorderWidth: 2
                });
            }
            
            new Chart(document.getElementById('cnnEmissionsChart'), {
                type: 'line',
                data: {
                    labels: cnnRuns.map((_, i) => `Run ${i + 1}`),
                    datasets: datasets
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        tooltip: {
                            callbacks: {
                                label: (context) => `${context.dataset.label}: ${context.parsed.y.toFixed(6)} g`
                            }
                        }
                    },
                    scales: {
                        y: {
                            beginAtZero: true,
                            title: { display: true, text: 'CO₂ Emissions (g)' }
                        }
                    }
                }
            });
        }

        // Transformer Emissions per Run
        if (transformerRuns.length > 0) {
            const codecarbonEmissions = transformerRuns.map(r => parseFloat(r.emissions_g) || 0);
            const carbontrackerEmissions = ctTransformerRuns.map(r => parseFloat(r.co2_g_adjusted) || 0);
            
            const datasets = [
                {
                    label: 'CodeCarbon',
                    data: codecarbonEmissions,
                    backgroundColor: 'rgba(102, 126, 234, 0.2)',
                    borderColor: 'rgba(102, 126, 234, 1)',
                    borderWidth: 3,
                    fill: true,
                    tension: 0.3,
                    pointRadius: 5,
                    pointHoverRadius: 7,
                    pointBackgroundColor: 'rgba(102, 126, 234, 1)',
                    pointBorderColor: '#fff',
                    pointBorderWidth: 2
                }
            ];
            
            if (carbontrackerEmissions.length > 0) {
                datasets.push({
                    label: 'CarbonTracker (PUE-adjusted)',
                    data: carbontrackerEmissions,
                    backgroundColor: 'rgba(255, 152, 0, 0.2)',
                    borderColor: 'rgba(255, 152, 0, 1)',
                    borderWidth: 3,
                    fill: true,
                    tension: 0.3,
                    pointRadius: 5,
                    pointHoverRadius: 7,
                    pointBackgroundColor: 'rgba(255, 152, 0, 1)',
                    pointBorderColor: '#fff',
                    pointBorderWidth: 2
                });
            }
            
            new Chart(document.getElementById('transformerEmissionsChart'), {
                type: 'line',
                data: {
                    labels: transformerRuns.map((_, i) => `Run ${i + 1}`),
                    datasets: datasets
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        tooltip: {
                            callbacks: {
                                label: (context) => `${context.dataset.label}: ${context.parsed.y.toFixed(6)} g`
                            }
                        }
                    },
                    scales: {
                        y: {
                            beginAtZero: true,
                            title: { display: true, text: 'CO₂ Emissions (g)' }
                        }
                    }
                }
            });
        }

        // ====================================================================
        // PERFORMANCE METRICS VISUALIZATIONS
        // ====================================================================

        const logregMetrics = parseCSV(logregMetricsCSV);
        const cnnMetrics = parseCSV(cnnMetricsCSV);
        const transformerMetrics = parseCSV(transformerMetricsCSV);

        console.log('LogReg metrics:', logregMetrics.length);
        console.log('CNN metrics:', cnnMetrics.length);
        console.log('Transformer metrics:', transformerMetrics.length);

        // Individual Model Performance Charts
        function createPerformanceChart(canvasId, metricsData, modelName, color) {
            if (!metricsData || metricsData.length === 0) return;

            const labels = metricsData.map((_, i) => `Run ${i + 1}`);
            const testAccuracy = metricsData.map(m => parseFloat(m.test_accuracy) * 100);
            const testF1 = metricsData.map(m => parseFloat(m.test_f1) * 100);

            new Chart(document.getElementById(canvasId), {
                type: 'line',
                data: {
                    labels: labels,
                    datasets: [
                        {
                            label: 'Performance - Accuracy',
                            data: testAccuracy,
                            borderColor: color,
                            backgroundColor: color.replace('1)', '0.1)'),
                            borderWidth: 3,
                            tension: 0.3,
                            pointRadius: 6,
                            pointHoverRadius: 8
                        },
                        {
                            label: 'Performance - F1 Score',
                            data: testF1,
                            borderColor: color.replace('rgba', 'rgba').replace('1)', '0.8)'),
                            backgroundColor: 'transparent',
                            borderWidth: 3,
                            tension: 0.3,
                            pointRadius: 6,
                            pointHoverRadius: 8,
                            pointStyle: 'rect'
                        }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: { display: true, position: 'top' },
                        tooltip: {
                            callbacks: {
                                label: (context) => `${context.dataset.label}: ${context.parsed.y.toFixed(2)}%`
                            }
                        }
                    },
                    scales: {
                        y: {
                            beginAtZero: false,
                            min: 80,
                            max: 94,
                            title: { display: true, text: 'Score (%)' }
                        }
                    }
                }
            });
        }

        createPerformanceChart('logregPerformanceChart', logregMetrics, 'Logistic Regression', 'rgba(118, 75, 162, 1)');
        createPerformanceChart('cnnPerformanceChart', cnnMetrics, 'CNN', 'rgba(237, 100, 166, 1)');
        createPerformanceChart('transformerPerformanceChart', transformerMetrics, 'Transformer', 'rgba(102, 126, 234, 1)');

        // Cross-Model Comparison Charts
        function getAverage(arr) {
            return arr.reduce((sum, val) => sum + val, 0) / arr.length;
        }

        const modelComparisonData = [];
        if (logregMetrics.length > 0) {
            modelComparisonData.push({
                name: 'LogReg',
                testAccuracy: getAverage(logregMetrics.map(m => parseFloat(m.test_accuracy) * 100)),
                testF1: getAverage(logregMetrics.map(m => parseFloat(m.test_f1) * 100)),
                color: 'rgba(118, 75, 162, 0.8)'
            });
        }
        if (cnnMetrics.length > 0) {
            modelComparisonData.push({
                name: 'CNN',
                testAccuracy: getAverage(cnnMetrics.map(m => parseFloat(m.test_accuracy) * 100)),
                testF1: getAverage(cnnMetrics.map(m => parseFloat(m.test_f1) * 100)),
                color: 'rgba(237, 100, 166, 0.8)'
            });
        }
        if (transformerMetrics.length > 0) {
            modelComparisonData.push({
                name: 'Transformer',
                testAccuracy: getAverage(transformerMetrics.map(m => parseFloat(m.test_accuracy) * 100)),
                testF1: getAverage(transformerMetrics.map(m => parseFloat(m.test_f1) * 100)),
                color: 'rgba(102, 126, 234, 0.8)'
            });
        }

        // Accuracy Comparison Chart
        if (modelComparisonData.length > 0) {
            new Chart(document.getElementById('accuracyComparisonChart'), {
                type: 'bar',
                data: {
                    labels: modelComparisonData.map(m => m.name),
                    datasets: [{
                        label: 'Average Test Accuracy',
                        data: modelComparisonData.map(m => m.testAccuracy),
                        backgroundColor: modelComparisonData.map(m => m.color),
                        borderColor: modelComparisonData.map(m => m.color.replace('0.8)', '1)')),
                        borderWidth: 2
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: { display: false },
                        tooltip: {
                            callbacks: {
                                label: (context) => `Accuracy: ${context.parsed.y.toFixed(2)}%`
                            }
                        }
                    },
                    scales: {
                        y: {
                            beginAtZero: false,
                            min: 75,
                            max: 95,
                            title: { display: true, text: 'Test Accuracy (%)' }
                        }
                    }
                }
            });

            // F1 Comparison Chart
            new Chart(document.getElementById('f1ComparisonChart'), {
                type: 'bar',
                data: {
                    labels: modelComparisonData.map(m => m.name),
                    datasets: [{
                        label: 'Average Test F1 Score',
                        data: modelComparisonData.map(m => m.testF1),
                        backgroundColor: modelComparisonData.map(m => m.color),
                        borderColor: modelComparisonData.map(m => m.color.replace('0.8)', '1)')),
                        borderWidth: 2
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: { display: false },
                        tooltip: {
                            callbacks: {
                                label: (context) => `F1 Score: ${context.parsed.y.toFixed(2)}%`
                            }
                        }
                    },
                    scales: {
                        y: {
                            beginAtZero: false,
                            min: 75,
                            max: 95,
                            title: { display: true, text: 'Test F1 Score (%)' }
                        }
                    }
                }
            });
        }

        // Emissions vs Performance Correlation Charts
        function mergeEmissionsAndMetrics(metricsData, modelName) {
            const projectName = `IMDB_${modelName}`;
            const emissionsForModel = data.filter(row => row.project_name === projectName);
            
            const merged = [];
            for (let i = 0; i < Math.min(metricsData.length, emissionsForModel.length); i++) {
                merged.push({
                    emissions: parseFloat(emissionsForModel[i].emissions_g),
                    testAccuracy: parseFloat(metricsData[i].test_accuracy) * 100,
                    testF1: parseFloat(metricsData[i].test_f1) * 100,
                    energy: parseFloat(emissionsForModel[i].energy_consumed_wh),
                    modelName: modelName
                });
            }
            return merged;
        }

        const allCorrelationData = [];
        if (logregMetrics.length > 0) {
            allCorrelationData.push(...mergeEmissionsAndMetrics(logregMetrics, 'LogReg'));
        }
        if (cnnMetrics.length > 0) {
            allCorrelationData.push(...mergeEmissionsAndMetrics(cnnMetrics, 'CNN'));
        }
        if (transformerMetrics.length > 0) {
            allCorrelationData.push(...mergeEmissionsAndMetrics(transformerMetrics, 'Transformer'));
        }

        const colorMap = {
            'LogReg': 'rgba(118, 75, 162, 0.7)',
            'CNN': 'rgba(237, 100, 166, 0.7)',
            'Transformer': 'rgba(102, 126, 234, 0.7)'
        };

        // Emissions vs Accuracy Scatter
        if (allCorrelationData.length > 0) {
            const groupedByModel = {};
            allCorrelationData.forEach(d => {
                if (!groupedByModel[d.modelName]) {
                    groupedByModel[d.modelName] = [];
                }
                groupedByModel[d.modelName].push({ x: d.emissions, y: d.testAccuracy });
            });

            const datasets = Object.keys(groupedByModel).map(model => ({
                label: model,
                data: groupedByModel[model],
                backgroundColor: colorMap[model],
                borderColor: colorMap[model].replace('0.7)', '1)'),
                pointRadius: 8,
                pointHoverRadius: 10
            }));

            new Chart(document.getElementById('emissionsVsAccuracyChart'), {
                type: 'scatter',
                data: { datasets: datasets },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: { display: true, position: 'top' },
                        tooltip: {
                            callbacks: {
                                label: (context) => `${context.dataset.label}: ${context.parsed.y.toFixed(2)}% @ ${context.parsed.x.toFixed(4)}g CO₂`
                            }
                        }
                    },
                    scales: {
                        x: {
                            title: { display: true, text: 'CO₂ Emissions (g)' }
                        },
                        y: {
                            beginAtZero: false,
                            min: 75,
                            max: 95,
                            title: { display: true, text: 'Test Accuracy (%)' }
                        }
                    }
                }
            });

            // Emissions vs F1 Scatter
            const f1GroupedByModel = {};
            allCorrelationData.forEach(d => {
                if (!f1GroupedByModel[d.modelName]) {
                    f1GroupedByModel[d.modelName] = [];
                }
                f1GroupedByModel[d.modelName].push({ x: d.emissions, y: d.testF1 });
            });

            const f1Datasets = Object.keys(f1GroupedByModel).map(model => ({
                label: model,
                data: f1GroupedByModel[model],
                backgroundColor: colorMap[model],
                borderColor: colorMap[model].replace('0.7)', '1)'),
                pointRadius: 8,
                pointHoverRadius: 10
            }));

            new Chart(document.getElementById('emissionsVsF1Chart'), {
                type: 'scatter',
                data: { datasets: f1Datasets },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: { display: true, position: 'top' },
                        tooltip: {
                            callbacks: {
                                label: (context) => `${context.dataset.label}: ${context.parsed.y.toFixed(2)}% @ ${context.parsed.x.toFixed(4)}g CO₂`
                            }
                        }
                    },
                    scales: {
                        x: {
                            title: { display: true, text: 'CO₂ Emissions (g)' }
                        },
                        y: {
                            beginAtZero: false,
                            min: 75,
                            max: 95,
                            title: { display: true, text: 'Test F1 Score (%)' }
                        }
                    }
                }
            });

            // Energy Efficiency Chart (Accuracy per Watt-hour)
            const efficiencyData = [];
            if (logregMetrics.length > 0) {
                const avgAccuracy = getAverage(logregMetrics.map(m => parseFloat(m.test_accuracy) * 100));
                const avgEnergy = getAverage(allCorrelationData.filter(d => d.modelName === 'LogReg').map(d => d.energy));
                efficiencyData.push({
                    name: 'LogReg',
                    efficiency: avgAccuracy / avgEnergy,
                    color: 'rgba(118, 75, 162, 0.8)'
                });
            }
            if (cnnMetrics.length > 0) {
                const avgAccuracy = getAverage(cnnMetrics.map(m => parseFloat(m.test_accuracy) * 100));
                const avgEnergy = getAverage(allCorrelationData.filter(d => d.modelName === 'CNN').map(d => d.energy));
                efficiencyData.push({
                    name: 'CNN',
                    efficiency: avgAccuracy / avgEnergy,
                    color: 'rgba(237, 100, 166, 0.8)'
                });
            }
            if (transformerMetrics.length > 0) {
                const avgAccuracy = getAverage(transformerMetrics.map(m => parseFloat(m.test_accuracy) * 100));
                const avgEnergy = getAverage(allCorrelationData.filter(d => d.modelName === 'Transformer').map(d => d.energy));
                efficiencyData.push({
                    name: 'Transformer',
                    efficiency: avgAccuracy / avgEnergy,
                    color: 'rgba(102, 126, 234, 0.8)'
                });
            }

            if (efficiencyData.length > 0) {
                new Chart(document.getElementById('efficiencyChart'), {
                    type: 'bar',
                    data: {
                        labels: efficiencyData.map(d => d.name),
                        datasets: [{
                            label: 'Efficiency (Accuracy % / Wh)',
                            data: efficiencyData.map(d => d.efficiency),
                            backgroundColor: efficiencyData.map(d => d.color),
                            borderColor: efficiencyData.map(d => d.color.replace('0.8)', '1)')),
                            borderWidth: 2
                        }]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        plugins: {
                            legend: { display: false },
                            tooltip: {
                                callbacks: {
                                    label: (context) => `Efficiency: ${context.parsed.y.toFixed(2)} accuracy%/Wh`
                                }
                            }
                        },
                        scales: {
                            y: {
                                beginAtZero: true,
                                title: { display: true, text: 'Accuracy % per Watt-hour' }
                            }
                        }
                    }
                });
            }
        }
    </script>
</body>
</html>'''
    
    # Replace placeholders with actual CSV data
    html_content = html_template.replace('CODECARBON_CSV_PLACEHOLDER', codecarbon_csv.replace('`', '\\`'))
    html_content = html_content.replace('CARBONTRACKER_CSV_PLACEHOLDER', carbontracker_csv.replace('`', '\\`'))
    html_content = html_content.replace('LOGREG_METRICS_CSV_PLACEHOLDER', logreg_metrics_csv.replace('`', '\\`'))
    html_content = html_content.replace('CNN_METRICS_CSV_PLACEHOLDER', cnn_metrics_csv.replace('`', '\\`'))
    html_content = html_content.replace('TRANSFORMER_METRICS_CSV_PLACEHOLDER', transformer_metrics_csv.replace('`', '\\`'))
    
    # Create reports directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Write HTML file
    with open(output_path, 'w') as f:
        f.write(html_content)
    
    print(f"Generated emissions report: {output_path}")
    print(f"Report contains {len(df_codecarbon)} CodeCarbon runs")
    if df_carbontracker is not None:
        print(f"Report contains {len(df_carbontracker)} CarbonTracker runs")
    print(f"Models: {', '.join(df_codecarbon['project_name'].unique())}")
    print(f"\nOpen the report with: open {output_path}")

if __name__ == '__main__':
    generate_html_report()
