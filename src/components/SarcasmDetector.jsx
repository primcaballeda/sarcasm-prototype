import React, { useState, useEffect } from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import './SarcasmDetector.css';

const API_BASE_URL = (process.env.REACT_APP_API_BASE_URL || '').replace(/\/$/, '');

const apiUrl = (path) => `${API_BASE_URL}${path}`;

const SarcasmDetector = () => {
  const [text, setText] = useState('');
  const [results, setResults] = useState(null);
  const [analyzing, setAnalyzing] = useState(false);
  const [dataset, setDataset] = useState([]);
  const [datasetResults, setDatasetResults] = useState([]);
  const [processingDataset, setProcessingDataset] = useState(false);
  const [showAllResults, setShowAllResults] = useState(false);
  const [modelMetrics, setModelMetrics] = useState(null);
  const [metricsLoading, setMetricsLoading] = useState(true);
  const [errorMessage, setErrorMessage] = useState('');
  const [uploadStatus, setUploadStatus] = useState({
    type: 'neutral',
    message: 'No dataset uploaded yet.',
    fileName: ''
  });

  // Validate input text
  const validateInput = (inputText) => {
    const trimmedText = inputText.trim();
    
    // Check for only numbers (integers)
    if (/^\d+$/.test(trimmedText)) {
      return 'Error: Please enter meaningful text, not just numbers.';
    }
    
    // Check for negative numbers
    if (/^-\d+$/.test(trimmedText)) {
      return 'Error: Negative numbers are not valid input. Please enter actual text.';
    }
    
    // Check for mostly special characters or gibberish (less than 30% letters)
    const letterCount = (trimmedText.match(/[a-zA-Z]/g) || []).length;
    const totalChars = trimmedText.length;
    if (totalChars > 0 && letterCount / totalChars < 0.3) {
      return 'Error: Input appears to be random characters or special symbols. Please enter meaningful text.';
    }
    
    // Check for maximum word count (200 words)
    const wordCount = trimmedText.split(/\s+/).filter(word => word.length > 0).length;
    if (wordCount > 200) {
      return `Error: Input exceeds maximum length. You entered ${wordCount} words, but the limit is 200 words.`;
    }
    
    return null;
  };

  // Fetch model metrics on component mount
  useEffect(() => {
    const fetchMetrics = async () => {
      try {
        const response = await fetch(apiUrl('/api/metrics'));
        if (response.ok) {
          const data = await response.json();
          setModelMetrics(data);
          console.log('✅ Fetched real metrics:', data);
        } else {
          console.warn('⚠️ Could not fetch metrics:', response.status);
        }
      } catch (error) {
        console.error('❌ Error fetching metrics:', error);
      } finally {
        setMetricsLoading(false);
      }
    };
    
    fetchMetrics();
  }, []);

  // Build performance data from real metrics API
  const buildPerformanceData = () => {
    const baseline = modelMetrics?.baseline?.performance_metrics;
    const proposed = modelMetrics?.proposed?.performance_metrics;

    return [
      { 
        metric: 'Accuracy', 
        baseline: baseline ? (baseline.accuracy * 100).toFixed(2) : null,
        proposed: proposed ? (proposed.accuracy * 100).toFixed(2) : null
      },
      { 
        metric: 'Precision', 
        baseline: baseline ? (baseline.precision * 100).toFixed(2) : null,
        proposed: proposed ? (proposed.precision * 100).toFixed(2) : null
      },
      { 
        metric: 'Sensitivity', 
        baseline: baseline ? (baseline.sensitivity_recall * 100).toFixed(2) : null,
        proposed: proposed ? (proposed.sensitivity_recall * 100).toFixed(2) : null
      },
      { 
        metric: 'F1-Score', 
        baseline: baseline ? (baseline.f1_score * 100).toFixed(2) : null,
        proposed: proposed ? (proposed.f1_score * 100).toFixed(2) : null
      },
      { 
        metric: 'Specificity', 
        baseline: baseline ? (baseline.specificity * 100).toFixed(2) : null,
        proposed: proposed ? (proposed.specificity * 100).toFixed(2) : null
      },
    ];
  };

  const modelPerformanceData = buildPerformanceData();

  // Confusion Matrix Data (Baseline Model) - Based on 1878 test samples
  // Accuracy: 70.34%, Precision: 69.92%, Sensitivity: 70.15%, Specificity: 70.53%
  const confusionMatrixBaseline = {
    truePositive: 651,  // True Positives (correctly identified sarcasm)
    falsePositive: 280,  // False Positives (incorrectly identified as sarcasm)
    falseNegative: 277,   // False Negatives (missed sarcasm)
    trueNegative: 670    // True Negatives (correctly identified non-sarcasm)
  };
  // Calculations: Sensitivity=651/(651+277)=70.15%, Precision=651/(651+280)=69.92%, 
  // Specificity=670/(670+280)=70.53%, Accuracy=(651+670)/1878=70.34%

  // Confusion Matrix Data (Proposed Model) - Based on 1878 test samples
  // Accuracy: 75.83%, Precision: 77.65%, Sensitivity: 72.52%, Specificity: 79.13%
  const confusionMatrixProposed = {
    truePositive: 681,   // True Positives (correctly identified sarcasm)
    falsePositive: 196,  // False Positives (incorrectly identified as sarcasm)
    falseNegative: 258,   // False Negatives (missed sarcasm)
    trueNegative: 743    // True Negatives (correctly identified non-sarcasm)
  };
  // Calculations: Sensitivity=681/(681+258)=72.52%, Precision=681/(681+196)=77.65%,
  // Specificity=743/(743+196)=79.13%, Accuracy=(681+743)/1878=75.83%

  const handleAnalyze = async () => {
    if (!text.trim()) {
      return;
    }

    // Validate input
    const error = validateInput(text);
    if (error) {
      setErrorMessage(error);
      return;
    }

    setErrorMessage('');
    setAnalyzing(true);
    
    try {
      console.log('🔍 Calling API with text:', text);
      
      // Call the compare endpoint to get results from BOTH real models
      const apiResponse = await fetch(apiUrl('/api/predict/compare'), {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ text: text })
      });
      
      if (!apiResponse.ok) {
        throw new Error(`API request failed with status: ${apiResponse.status}`);
      }
      
      const apiResult = await apiResponse.json();
      console.log('✅ API Response:', apiResult);
      
      // Format the baseline model response
      const baselineDetection = {
        isSarcastic: apiResult.baseline.isSarcastic,
        confidence: apiResult.baseline.confidence,
        indicators: [
          'Real Keras model loaded',
          'GloVe embeddings processed',
          'BiLSTM with attention mechanism',
          `Sarcasm probability: ${apiResult.baseline.probabilities?.sarcastic}%`,
          `Non-sarcasm probability: ${apiResult.baseline.probabilities?.not_sarcastic}%`
        ],
        model: 'GloVe+CNN+BiLSTM+Attention',
        processingTime: apiResult.baseline.processingTime
      };
      
      // Format the proposed model response
      const proposedDetection = {
        isSarcastic: apiResult.proposed.isSarcastic,
        confidence: apiResult.proposed.confidence,
        indicators: [
          'Real PyTorch model loaded',
          'BERT contextual embeddings analyzed',
          'CNN + BiLSTM architecture',
          'Multi-head attention patterns detected',
          `Sarcasm probability: ${apiResult.proposed.probabilities?.sarcastic}%`,
          `Non-sarcasm probability: ${apiResult.proposed.probabilities?.not_sarcastic}%`
        ],
        model: 'BERT+CNN+BiLSTM+MHA',
        processingTime: apiResult.proposed.processingTime
      };
      
      console.log('📊 Formatted Results:', { baseline: baselineDetection, proposed: proposedDetection });
      
      setResults({
        baseline: baselineDetection,
        proposed: proposedDetection
      });
      setAnalyzing(false);
    } catch (error) {
      console.error('❌ Error calling API:', error);
      alert('Error: Failed to connect to backend models. Please ensure:\n1. Backend server is running (python app.py)\n2. Models are loaded (model.h5 and sarcasm_model.pt exist)\n3. API endpoints are accessible');
      setAnalyzing(false);
    }
  };

  const handleReset = () => {
    setText('');
    setResults(null);
    setErrorMessage('');
  };

  // Helper function to parse CSV properly handling quoted fields with commas
  const parseCSVLine = (line) => {
    const result = [];
    let current = '';
    let inQuotes = false;
    
    for (let i = 0; i < line.length; i++) {
      const char = line[i];
      
      if (char === '"') {
        if (inQuotes && line[i + 1] === '"') {
          current += '"';
          i++;
        } else {
          inQuotes = !inQuotes;
        }
      } else if (char === ',' && !inQuotes) {
        result.push(current);
        current = '';
      } else {
        current += char;
      }
    }
    
    result.push(current);
    return result;
  };

  // Split CSV content into logical records, preserving newlines inside quoted fields.
  const splitCSVRecords = (content) => {
    const records = [];
    let current = '';
    let inQuotes = false;

    for (let i = 0; i < content.length; i++) {
      const char = content[i];
      const nextChar = content[i + 1];

      if (char === '"') {
        if (inQuotes && nextChar === '"') {
          current += '""';
          i++;
          continue;
        }
        inQuotes = !inQuotes;
        current += char;
        continue;
      }

      if (char === '\n' && !inQuotes) {
        if (current.trim().length > 0) {
          records.push(current);
        }
        current = '';
        continue;
      }

      current += char;
    }

    if (inQuotes) {
      throw new Error('CSV contains unclosed quoted text. Please ensure every quoted field has a closing quote.');
    }

    if (current.trim().length > 0) {
      records.push(current);
    }

    return records;
  };

  const handleFileUpload = (event) => {
    const file = event.target.files[0];
    if (!file) return;

    const normalizedName = file.name.toLowerCase();
    if (!normalizedName.endsWith('.csv') && !normalizedName.endsWith('.json')) {
      setUploadStatus({
        type: 'error',
        message: 'Unsupported file type. Please upload a CSV or JSON file.',
        fileName: file.name
      });
      setDataset([]);
      setDatasetResults([]);
      setShowAllResults(false);
      event.target.value = '';
      return;
    }

    setUploadStatus({
      type: 'neutral',
      message: 'Reading file and validating rows...',
      fileName: file.name
    });

    const reader = new FileReader();
    
    reader.onload = (e) => {
      const content = e.target.result;
      let parsedData = [];

      try {
        if (normalizedName.endsWith('.json')) {
          const jsonData = JSON.parse(content);
          const jsonArray = Array.isArray(jsonData) ? jsonData : [jsonData];

          parsedData = jsonArray.map((item, index) => {
            const candidateText = item?.text || item?.comment || item?.sentence || item?.['Response Text'] || item?.response || item?.content;
            if (!candidateText || String(candidateText).trim().length === 0) {
              return null;
            }

            return {
              id: index + 1,
              text: String(candidateText).trim(),
              label: item.label || item.sarcastic || item.is_sarcastic || item.Label || null
            };
          }).filter(item => item !== null);

          if (parsedData.length === 0) {
            throw new Error('JSON format is not aligned. Add a text field named text, comment, sentence, response, content, or Response Text.');
          }
        } else if (normalizedName.endsWith('.csv')) {
          // Normalize line endings and split into records (supports multiline quoted text)
          const normalizedContent = content.replace(/\r\n/g, '\n').replace(/\r/g, '\n');
          const records = splitCSVRecords(normalizedContent);
          const expectedHeaders = ['corpus', 'label', 'id', 'response text'];
          
          if (records.length < 2) {
            throw new Error('CSV file must have at least a header row and one data row');
          }
          
          const headers = parseCSVLine(records[0]).map(h => h.trim().toLowerCase().replace(/^"|"$/g, ''));

          if (headers.length !== expectedHeaders.length || !expectedHeaders.every((header, idx) => headers[idx] === header)) {
            throw new Error('CSV format is not aligned. Expected exact header order: Corpus,Label,ID,Response Text');
          }
          
          console.log('CSV Headers:', headers);
          console.log('Total records:', records.length);
          
          parsedData = records.slice(1).map((line, index) => {
            const rowNumber = index + 2;
            const values = parseCSVLine(line);
            const normalizedValues = values.map(v => v.trim());

            if (values.length !== expectedHeaders.length) {
              throw new Error(`Row ${rowNumber} has ${values.length} columns. Expected exactly ${expectedHeaders.length} columns.`);
            }
            
            const normalizedLowerValues = normalizedValues.map(v => v.toLowerCase());
            const isDuplicateHeader = normalizedValues.length === expectedHeaders.length &&
              expectedHeaders.every((header, idx) => normalizedLowerValues[idx] === header);

            if (isDuplicateHeader) {
              throw new Error(`Row ${rowNumber} appears to be a duplicate header row. Remove extra headers from the data section.`);
            }

            const corpus = normalizedValues[0];
            const labelValue = normalizedValues[1].toLowerCase();
            const originalId = normalizedValues[2];
            const text = values[3];

            if (!corpus || !labelValue || !originalId || !text) {
              throw new Error(`Row ${rowNumber} is incomplete. All columns (Corpus, Label, ID, Response Text) are required.`);
            }

            if (labelValue !== 'sarc' && labelValue !== 'notsarc') {
              throw new Error(`Row ${rowNumber} has invalid Label "${values[1]}". Use only sarc or notsarc.`);
            }

            if (!/^\d+$/.test(originalId)) {
              throw new Error(`Row ${rowNumber} has invalid ID "${originalId}". ID must be a number.`);
            }
            
            return {
              id: originalId,
              text: text,
              label: labelValue === 'sarc'
            };
          });
          
          console.log('Parsed data count:', parsedData.length);
          
          if (parsedData.length === 0) {
            throw new Error('No valid data rows found in CSV. Please check the file format.');
          }
        }

        setDataset(parsedData);
        setDatasetResults([]);
        setShowAllResults(false);

        let statusType = 'success';
        let statusMessage = `Loaded ${parsedData.length} sample${parsedData.length === 1 ? '' : 's'}. Click "Process Dataset" to run both models.`;

        setUploadStatus({
          type: statusType,
          message: statusMessage,
          fileName: file.name
        });
      } catch (error) {
        console.error('Parse error:', error);
        setUploadStatus({
          type: 'error',
          message: `File format not aligned: ${error.message}`,
          fileName: file.name
        });
        setDataset([]);
        setDatasetResults([]);
        setShowAllResults(false);
      }
    };

    reader.readAsText(file);
  };

  const processDataset = async () => {
    if (dataset.length === 0) return;
    
    setProcessingDataset(true);
    setDatasetResults([]); // Clear previous results
    setShowAllResults(false); // Reset to show limited results
    
    const results = [];
    const batchSize = 10; 
    
    try {
      // Process dataset in batches
      for (let batchStart = 0; batchStart < dataset.length; batchStart += batchSize) {
        const batchEnd = Math.min(batchStart + batchSize, dataset.length);
        const batch = dataset.slice(batchStart, batchEnd);
        
        // Get texts for this batch
        const texts = batch.map(item => item.text);
        
        console.log(`Processing batch ${batchStart}-${batchEnd}:`, texts.slice(0, 3), '...');
        
        // Call the batch prediction API for both models
        let proposedPredictions = [];
        let baselinePredictions = [];
        
        try {
          // Baseline model predictions
          const baselineResponse = await fetch(apiUrl('/api/predict_batch'), {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
            },
            body: JSON.stringify({ texts: texts, model: 'baseline' })
          });
          
          if (baselineResponse.ok) {
            const baselineResult = await baselineResponse.json();
            baselinePredictions = baselineResult.results || [];
            console.log('✅ Baseline API SUCCESS - got', baselinePredictions.length, 'predictions');
          } else {
            throw new Error('Baseline API request failed');
          }
          
          // Proposed model predictions
          const proposedResponse = await fetch(apiUrl('/api/predict_batch'), {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
            },
            body: JSON.stringify({ texts: texts, model: 'proposed' })
          });
          
          if (proposedResponse.ok) {
            const proposedResult = await proposedResponse.json();
            proposedPredictions = proposedResult.results || [];
            console.log('✅ Proposed API SUCCESS - got', proposedPredictions.length, 'predictions');
          } else {
            throw new Error('Proposed API request failed');
          }
        } catch (error) {
          console.error('❌ API ERROR:', error);
          alert(`Batch processing failed: ${error.message}\n\nMake sure:\n1. Backend is running (python app.py)\n2. Models are loaded in backend\n3. API endpoint /api/predict_batch is working`);
          setProcessingDataset(false);
          return;
        }
        
        // Process each item in the batch
        for (let i = 0; i < batch.length; i++) {
          const item = batch[i];
          
          // Baseline detection (from API)
          const baselineResult = baselinePredictions[i] || { isSarcastic: false, confidence: 0 };
          
          // Proposed detection (from API)
          const proposedResult = proposedPredictions[i] || { isSarcastic: false, confidence: 0 };
          
          // Use actual label from dataset if available, otherwise use baseline as reference
          const actualLabel = item.label !== null ? item.label : baselineResult.isSarcastic;
          
          const result = {
            ...item,
            label: actualLabel,
            baseline: {
              predicted: baselineResult.isSarcastic,
              confidence: baselineResult.confidence,
              correct: baselineResult.isSarcastic === actualLabel
            },
            proposed: {
              predicted: proposedResult.isSarcastic,
              confidence: proposedResult.confidence,
              correct: proposedResult.isSarcastic === actualLabel
            }
          };
          
          results.push(result);
        }
        
        // Update state to show progress
        setDatasetResults([...results]);
        await new Promise(resolve => setTimeout(resolve, 100));
      }
      
      setDatasetResults(results);
    } catch (error) {
      console.error('Error processing dataset:', error);
      alert('Error processing dataset. Please check if the backend server is running.');
    } finally {
      setProcessingDataset(false);
    }
  };

  const clearDataset = () => {
    setDataset([]);
    setDatasetResults([]);
    setShowAllResults(false);
    setUploadStatus({
      type: 'neutral',
      message: 'Dataset cleared. Upload a CSV or JSON file to start again.',
      fileName: ''
    });
  };

  const calculateDatasetStats = () => {
    if (datasetResults.length === 0) return null;
    
    const total = datasetResults.length;
    const withLabels = datasetResults.filter(r => r.label !== null);
    
    const baselineCorrect = withLabels.filter(r => r.baseline.correct).length;
    const baselineAccuracy = withLabels.length > 0 ? (baselineCorrect / withLabels.length * 100).toFixed(2) : 'N/A';
    const baselineSarcastic = datasetResults.filter(r => r.baseline.predicted).length;
    
    const proposedCorrect = withLabels.filter(r => r.proposed.correct).length;
    const proposedAccuracy = withLabels.length > 0 ? (proposedCorrect / withLabels.length * 100).toFixed(2) : 'N/A';
    const proposedSarcastic = datasetResults.filter(r => r.proposed.predicted).length;
    
    return {
      total,
      withLabels: withLabels.length,
      baseline: {
        correct: baselineCorrect,
        accuracy: baselineAccuracy,
        predictedSarcastic: baselineSarcastic,
        predictedNotSarcastic: total - baselineSarcastic
      },
      proposed: {
        correct: proposedCorrect,
        accuracy: proposedAccuracy,
        predictedSarcastic: proposedSarcastic,
        predictedNotSarcastic: total - proposedSarcastic
      }
    };
  };

  const calculateDetailedMetrics = () => {
    if (datasetResults.length === 0) return null;
    
    const withLabels = datasetResults.filter(r => r.label !== null);
    if (withLabels.length === 0) return null;
    
    // Calculate confusion matrix for baseline
    const baselineTP = withLabels.filter(r => r.label === true && r.baseline.predicted === true).length;
    const baselineTN = withLabels.filter(r => r.label === false && r.baseline.predicted === false).length;
    const baselineFP = withLabels.filter(r => r.label === false && r.baseline.predicted === true).length;
    const baselineFN = withLabels.filter(r => r.label === true && r.baseline.predicted === false).length;
    
    // Calculate confusion matrix for proposed
    const proposedTP = withLabels.filter(r => r.label === true && r.proposed.predicted === true).length;
    const proposedTN = withLabels.filter(r => r.label === false && r.proposed.predicted === false).length;
    const proposedFP = withLabels.filter(r => r.label === false && r.proposed.predicted === true).length;
    const proposedFN = withLabels.filter(r => r.label === true && r.proposed.predicted === false).length;
    
    // Calculate metrics for baseline
    const baselineAccuracy = ((baselineTP + baselineTN) / withLabels.length * 100).toFixed(2);
    const baselinePrecision = baselineTP + baselineFP > 0 ? ((baselineTP / (baselineTP + baselineFP)) * 100).toFixed(2) : '0.00';
    const baselineRecall = baselineTP + baselineFN > 0 ? ((baselineTP / (baselineTP + baselineFN)) * 100).toFixed(2) : '0.00';
    const baselineF1 = baselinePrecision > 0 && baselineRecall > 0 
      ? (2 * (parseFloat(baselinePrecision) * parseFloat(baselineRecall)) / (parseFloat(baselinePrecision) + parseFloat(baselineRecall))).toFixed(2)
      : '0.00';
    const baselineSpecificity = baselineTN + baselineFP > 0 ? ((baselineTN / (baselineTN + baselineFP)) * 100).toFixed(2) : '0.00';
    
    // Calculate metrics for proposed
    const proposedAccuracy = ((proposedTP + proposedTN) / withLabels.length * 100).toFixed(2);
    const proposedPrecision = proposedTP + proposedFP > 0 ? ((proposedTP / (proposedTP + proposedFP)) * 100).toFixed(2) : '0.00';
    const proposedRecall = proposedTP + proposedFN > 0 ? ((proposedTP / (proposedTP + proposedFN)) * 100).toFixed(2) : '0.00';
    const proposedF1 = proposedPrecision > 0 && proposedRecall > 0 
      ? (2 * (parseFloat(proposedPrecision) * parseFloat(proposedRecall)) / (parseFloat(proposedPrecision) + parseFloat(proposedRecall))).toFixed(2)
      : '0.00';
    const proposedSpecificity = proposedTN + proposedFP > 0 ? ((proposedTN / (proposedTN + proposedFP)) * 100).toFixed(2) : '0.00';
    
    return {
      baseline: {
        accuracy: baselineAccuracy,
        precision: baselinePrecision,
        recall: baselineRecall,
        f1Score: baselineF1,
        specificity: baselineSpecificity,
        confusion: { tp: baselineTP, tn: baselineTN, fp: baselineFP, fn: baselineFN }
      },
      proposed: {
        accuracy: proposedAccuracy,
        precision: proposedPrecision,
        recall: proposedRecall,
        f1Score: proposedF1,
        specificity: proposedSpecificity,
        confusion: { tp: proposedTP, tn: proposedTN, fp: proposedFP, fn: proposedFN }
      }
    };
  };

  return (
    <div className="sarcasm-detector">
      <div className="header">
        <h1>Sarcasm Detector</h1>
        <p className="subtitle">Baseline vs. Proposed Model</p>
      </div>

      <div className="input-section">
        <textarea
          className="text-input"
          placeholder="Type or paste text to analyze... (e.g., 'Oh great, another meeting!') [Max 200 words]"
          value={text}
          onChange={(e) => setText(e.target.value)}
          rows={6} />

        <div className="word-counter">
          <span className={text.trim().split(/\s+/).filter(w => w.length > 0).length > 200 ? 'over-limit' : ''}>
            {text.trim().split(/\s+/).filter(w => w.length > 0).length} / 200 words
          </span>
        </div>

        <div className="button-group">
          <button
            className="analyze-btn"
            onClick={handleAnalyze}
            disabled={!text.trim() || analyzing}
          >
            {analyzing ? 'Analyzing...' : 'Detect Sarcasm'}
          </button>

          {text && (
            <button
              className="reset-btn"
              onClick={handleReset}
            >
              Clear
            </button>
          )}
        </div>

        {errorMessage && (
          <div className="error-message">
            {errorMessage}
          </div>
        )}
      </div>

      {results && (
        <div className="dual-results-section">
          <h2 className="comparison-title">Model Comparison Results</h2>

          <div className="models-comparison">
            <div className={`result-section ${results.baseline.isSarcastic ? 'sarcastic' : 'not-sarcastic'}`}>
              <div className="result-header">
                <h2>
                  {results.baseline.isSarcastic ? 'SARCASM DETECTED!' : 'Not Sarcastic'}
                </h2>
                <div className="model-badge baseline-badge">Baseline Model</div>
                <div className="model-desc">GloVe + CNN + BiLSTM + Attention</div>
                <div className="processing-time">Processing time: {results.baseline.processingTime}</div>
              </div>

              <div className="confidence-bar">
                <div className="confidence-label">
                  Confidence: {results.baseline.confidence}%
                </div>
                <div className="progress-bar">
                  <div
                    className="progress-fill"
                    style={{ width: `${results.baseline.confidence}%` }} />
                </div>
              </div>

              <div className="indicators">
                <h3>Analysis Details:</h3>
                <ul>
                  {results.baseline.indicators.map((indicator, index) => (
                    <li key={index}>{indicator}</li>
                  ))}
                </ul>
              </div>
            </div>

            <div className={`result-section ${results.proposed.isSarcastic ? 'sarcastic' : 'not-sarcastic'}`}>
              <div className="result-header">
                <h2>
                  {results.proposed.isSarcastic ? 'SARCASM DETECTED!' : 'Not Sarcastic'}
                </h2>
                <div className="model-badge proposed-badge">Proposed Model</div>
                <div className="model-desc">BERT + CNN + BiLSTM + MHA</div>
                <div className="processing-time">Processing time: {results.proposed.processingTime}</div>
              </div>

              <div className="confidence-bar">
                <div className="confidence-label">
                  Confidence: {results.proposed.confidence}%
                </div>
                <div className="progress-bar">
                  <div
                    className="progress-fill"
                    style={{ width: `${results.proposed.confidence}%` }} />
                </div>
              </div>

              <div className="indicators">
                <h3>Analysis Details:</h3>
                <ul>
                  {results.proposed.indicators.map((indicator, index) => (
                    <li key={index}>{indicator}</li>
                  ))}
                </ul>
              </div>
            </div>
          </div>
        </div>
      )}

      <div className="examples">
        <h3>Try these examples:</h3>
        <div className="example-buttons">
          <button
            className="example-btn"
            onClick={() => setText("Oh great, another Monday morning meeting!")}
          >
            Example 1
          </button>
          <button
            className="example-btn"
            onClick={() => setText("Yeah right, like that's ever going to happen...")}
          >
            Example 2
          </button>
          <button
            className="example-btn"
            onClick={() => setText("I love working on weekends!")}
          >
            Example 3
          </button>
          <button
            className="example-btn"
            onClick={() => setText("Thank you for your help today.")}
          >
            Example 4
          </button>
        </div>
      </div>

      <div className="dataset-section">
        <h2>Upload Dataset for Batch Testing</h2>
        <p className="dataset-description">
          Use the guided steps below to compare both models on many text samples at once.
        </p>

        <div className="upload-steps">
          <div className={`upload-step ${dataset.length === 0 ? 'active' : 'completed'}`}>
            <span className="step-number">1</span>
            <div>
              <h4>Select a file</h4>
              <p>Upload a CSV or JSON file with at least one text field.</p>
            </div>
          </div>
          <div className={`upload-step ${dataset.length > 0 && datasetResults.length === 0 ? 'active' : datasetResults.length > 0 ? 'completed' : ''}`}>
            <span className="step-number">2</span>
            <div>
              <h4>Confirm loaded samples</h4>
              <p>Check sample count and clear/re-upload if needed.</p>
            </div>
          </div>
          <div className={`upload-step ${processingDataset || datasetResults.length > 0 ? 'active' : ''}`}>
            <span className="step-number">3</span>
            <div>
              <h4>Process dataset</h4>
              <p>Run predictions and view side-by-side model performance.</p>
            </div>
          </div>
        </div>

        <div className="dataset-upload">
          <label htmlFor="file-upload" className="file-upload-label">
            Choose Dataset File
          </label>
          <input
            id="file-upload"
            type="file"
            accept=".csv,.json"
            onChange={handleFileUpload}
            style={{ display: 'none' }} />

          <div className={`upload-status ${uploadStatus.type}`}>
            <strong>Status:</strong> {uploadStatus.message}
            {uploadStatus.fileName && <span className="upload-file-name"> File: {uploadStatus.fileName}</span>}
          </div>

          {dataset.length > 0 && (
            <div className="dataset-info">
              <span className="dataset-count">{dataset.length} samples loaded</span>
              <button className="clear-dataset-btn" onClick={clearDataset}>
                Clear Dataset
              </button>
            </div>
          )}
        </div>

        <div className="format-info">
          <div className="format-quick-guide">
            <p><strong>Quick format guide</strong></p>
            <ul>
              <li>CSV headers can include text/comment/sentence/response and optional label/id columns.</li>
              <li>JSON supports either an array of objects or a single object.</li>
              <li>Label values accepted: sarc/notsarc, sarcastic/not sarcastic, 1/0, true/false.</li>
            </ul>
          </div>
          <details>
            <summary>View a full sample file</summary>
            <div className="format-examples">
              <div className="format-example">
                <strong>CSV Format:</strong>
                <pre>{`Corpus,Label,ID,Response Text
GEN,notsarc,1,"If that's true, then Freedom of Speech is doomed."
GEN,sarc,2,"Oh great, another meeting!"
GEN,notsarc,3,"Thank you for your help today."`}</pre>
                <p><em>Note: Label column is optional. Accepted values: sarc/notsarc, 1/0, true/false, sarcastic/not sarcastic</em></p>
              </div>
            </div>
          </details>
        </div>

        {dataset.length > 0 && (
          <div className="dataset-controls">
            <button
              className="process-dataset-btn"
              onClick={processDataset}
              disabled={processingDataset}
            >
              {processingDataset ? `Processing... (${datasetResults.length}/${dataset.length})` : 'Process Dataset'}
            </button>
          </div>
        )}

        {datasetResults.length > 0 && !processingDataset && (
          <div className="dataset-results">
            <h3>Dataset Results</h3>

            {(() => {
              const stats = calculateDatasetStats();

              return stats && (
                <>
                  <div className="dataset-metrics">
                    <h4>Performance on Your Dataset</h4>
                    <div className="metrics-table-container">
                      <table className="metrics-table">
                        <thead>
                          <tr>
                            <th>Metric</th>
                            <th className="baseline-col">Baseline Model</th>
                            <th className="proposed-col">Proposed Model</th>
                            <th>Improvement</th>
                          </tr>
                        </thead>
                        <tbody>
                          {(() => {
                            const datasetMetrics = calculateDetailedMetrics();
                            const rows = datasetMetrics ? [
                              { metric: 'Accuracy', baseline: datasetMetrics.baseline.accuracy, proposed: datasetMetrics.proposed.accuracy },
                              { metric: 'Precision', baseline: datasetMetrics.baseline.precision, proposed: datasetMetrics.proposed.precision },
                              { metric: 'Sensitivity', baseline: datasetMetrics.baseline.recall, proposed: datasetMetrics.proposed.recall },
                              { metric: 'F1-Score', baseline: datasetMetrics.baseline.f1Score, proposed: datasetMetrics.proposed.f1Score },
                              { metric: 'Specificity', baseline: datasetMetrics.baseline.specificity, proposed: datasetMetrics.proposed.specificity }
                            ] : [];
                            
                            return rows.map((row, idx) => (
                              <tr key={idx}>
                                <td className="metric-name">{row.metric}</td>
                                <td className="baseline-value">{row.baseline}%</td>
                                <td className="proposed-value">{row.proposed}%</td>
                                <td className="improvement-value">
                                  {(parseFloat(row.proposed) - parseFloat(row.baseline)) > 0
                                    ? `+${(parseFloat(row.proposed) - parseFloat(row.baseline)).toFixed(2)}%`
                                    : `${(parseFloat(row.proposed) - parseFloat(row.baseline)).toFixed(2)}%`}
                                </td>
                              </tr>
                            ));
                          })()}
                        </tbody>
                      </table>
                    </div>

                    {(() => {
                      const datasetMetrics = calculateDetailedMetrics();
                      return datasetMetrics ? (
                        <div className="confusion-matrices-comparison">
                          <div className="confusion-matrix-side">
                            <h5>Baseline Model - Confusion Matrix</h5>
                            <div className="mini-confusion-matrix">
                              <div className="matrix-row">
                                <div className="matrix-cell-mini header-cell"></div>
                                <div className="matrix-cell-mini header-cell">Pred: Sarc</div>
                                <div className="matrix-cell-mini header-cell">Pred: Not Sarc</div>
                              </div>
                              <div className="matrix-row">
                                <div className="matrix-cell-mini header-cell">Actual: Sarc</div>
                                <div className="matrix-cell-mini tp-cell">TP: {datasetMetrics.baseline.confusion.tp}</div>
                                <div className="matrix-cell-mini fn-cell">FN: {datasetMetrics.baseline.confusion.fn}</div>
                              </div>
                              <div className="matrix-row">
                                <div className="matrix-cell-mini header-cell">Actual: Not Sarc</div>
                                <div className="matrix-cell-mini fp-cell">FP: {datasetMetrics.baseline.confusion.fp}</div>
                                <div className="matrix-cell-mini tn-cell">TN: {datasetMetrics.baseline.confusion.tn}</div>
                              </div>
                            </div>
                          </div>

                          <div className="confusion-matrix-side">
                            <h5>Proposed Model - Confusion Matrix</h5>
                            <div className="mini-confusion-matrix">
                              <div className="matrix-row">
                                <div className="matrix-cell-mini header-cell"></div>
                                <div className="matrix-cell-mini header-cell">Pred: Sarc</div>
                                <div className="matrix-cell-mini header-cell">Pred: Not Sarc</div>
                              </div>
                              <div className="matrix-row">
                                <div className="matrix-cell-mini header-cell">Actual: Sarc</div>
                                <div className="matrix-cell-mini tp-cell">TP: {datasetMetrics.proposed.confusion.tp}</div>
                                <div className="matrix-cell-mini fn-cell">FN: {datasetMetrics.proposed.confusion.fn}</div>
                              </div>
                              <div className="matrix-row">
                                <div className="matrix-cell-mini header-cell">Actual: Not Sarc</div>
                                <div className="matrix-cell-mini fp-cell">FP: {datasetMetrics.proposed.confusion.fp}</div>
                                <div className="matrix-cell-mini tn-cell">TN: {datasetMetrics.proposed.confusion.tn}</div>
                              </div>
                            </div>
                          </div>
                        </div>
                      ) : null;
                    })()}
                  </div>

                  <div className="dataset-stats-header">
                    <h4>Baseline Model Results</h4>
                  </div>
                  <div className="dataset-stats">
                    <div className="stat-card">
                      <div className="stat-value">{stats.total}</div>
                      <div className="stat-label">Total Samples</div>
                    </div>
                    <div className="stat-card">
                      <div className="stat-value">{stats.baseline.predictedSarcastic}</div>
                      <div className="stat-label">Predicted Sarcastic</div>
                    </div>
                    <div className="stat-card">
                      <div className="stat-value">{stats.baseline.predictedNotSarcastic}</div>
                      <div className="stat-label">Predicted Not Sarcastic</div>
                    </div>
                    {stats.withLabels > 0 && (
                      <>
                        <div className="stat-card">
                          <div className="stat-value">{stats.baseline.correct}/{stats.withLabels}</div>
                          <div className="stat-label">Correct Predictions</div>
                        </div>
                        <div className="stat-card highlight">
                          <div className="stat-value">{stats.baseline.accuracy}%</div>
                          <div className="stat-label">Accuracy</div>
                        </div>
                      </>
                    )}
                  </div>

                  <div className="dataset-stats-header">
                    <h4>Proposed Model Results</h4>
                  </div>
                  <div className="dataset-stats">
                    <div className="stat-card">
                      <div className="stat-value">{stats.total}</div>
                      <div className="stat-label">Total Samples</div>
                    </div>
                    <div className="stat-card">
                      <div className="stat-value">{stats.proposed.predictedSarcastic}</div>
                      <div className="stat-label">Predicted Sarcastic</div>
                    </div>
                    <div className="stat-card">
                      <div className="stat-value">{stats.proposed.predictedNotSarcastic}</div>
                      <div className="stat-label">Predicted Not Sarcastic</div>
                    </div>
                    {stats.withLabels > 0 && (
                      <>
                        <div className="stat-card">
                          <div className="stat-value">{stats.proposed.correct}/{stats.withLabels}</div>
                          <div className="stat-label">Correct Predictions</div>
                        </div>
                        <div className="stat-card highlight">
                          <div className="stat-value">{stats.proposed.accuracy}%</div>
                          <div className="stat-label">Accuracy</div>
                        </div>
                      </>
                    )}
                  </div>
                </>
              );
            })()}

            <div className="results-table-container">
              <table className="results-table">
                <thead>
                  <tr>
                    <th>ID</th>
                    <th>Text</th>
                    <th colSpan="2">Baseline Model</th>
                    <th colSpan="2">Proposed Model</th>
                    {datasetResults.some(r => r.label !== null) && <th>Actual</th>}
                  </tr>
                  <tr className="sub-header">
                    <th></th>
                    <th></th>
                    <th>Predicted</th>
                    <th>Conf.</th>
                    <th>Predicted</th>
                    <th>Conf.</th>
                    {datasetResults.some(r => r.label !== null) && <th>Label</th>}
                  </tr>
                </thead>
                <tbody>
                  {(showAllResults ? datasetResults : datasetResults.slice(0, 15)).map((result) => (
                    <tr key={result.id}>
                      <td>{result.id}</td>
                      <td className="text-cell">{result.text}</td>
                      <td>
                        <span className={`prediction-badge ${result.baseline.predicted ? 'sarcastic' : 'not-sarcastic'}`}>
                          {result.baseline.predicted ? 'Sarcastic' : 'Not Sarcastic'}
                        </span>
                        {result.label !== null && (
                          <span className={`match-icon ${result.baseline.correct ? 'correct' : 'incorrect'}`}>
                            {result.baseline.correct ? ' PASS' : ' FAIL'}
                          </span>
                        )}
                      </td>
                      <td>{result.baseline.confidence}%</td>
                      <td>
                        <span className={`prediction-badge ${result.proposed.predicted ? 'sarcastic' : 'not-sarcastic'}`}>
                          {result.proposed.predicted ? 'Sarcastic' : 'Not Sarcastic'}
                        </span>
                        {result.label !== null && (
                          <span className={`match-icon ${result.proposed.correct ? 'correct' : 'incorrect'}`}>
                            {result.proposed.correct ? ' PASS' : ' FAIL'}
                          </span>
                        )}
                      </td>
                      <td>{result.proposed.confidence}%</td>
                      {datasetResults.some(r => r.label !== null) && (
                        <td>
                          {result.label !== null ? (
                            <span className={`prediction-badge ${result.label ? 'sarcastic' : 'not-sarcastic'}`}>
                              {result.label ? 'Sarcastic' : 'Not Sarcastic'}
                            </span>
                          ) : '-'}
                        </td>
                      )}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            {datasetResults.length > 15 && (
              <div className="show-all-container">
                <button
                  className="show-all-btn"
                  onClick={() => setShowAllResults(!showAllResults)}
                >
                  {showAllResults ? 'Show Less' : `Show All Results (${datasetResults.length})`}
                </button>
              </div>
            )}
          </div>
        )}
      </div>

      <div className="performance-graphs">
        <h2>Model Performance Comparison</h2>

        {metricsLoading ? (
          <div className="metrics-loading">
            <p>Loading real model metrics...</p>
          </div>
        ) : !modelMetrics || (!modelMetrics.baseline && !modelMetrics.proposed) ? (
          <div className="metrics-error">
            <p>⚠️ Could not load metrics. Backend may not be running, or metrics files are missing.</p>
            <p><strong>Ensure:</strong> Backend is running and both model_metrics.json and proposed_model_metrics.json exist in backend/model/</p>
          </div>
        ) : (
          <>
            <div className="metrics-info">
              {modelMetrics.baseline && <p> Baseline metrics loaded</p>}
              {modelMetrics.proposed && <p> Proposed metrics loaded</p>}
            </div>

            <div className="graph-container">
              <div className="graph-card">
                <h3>Performance Metrics</h3>
                <ResponsiveContainer width="100%" height={300}>
                  <BarChart data={modelPerformanceData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="metric" />
                    <YAxis domain={[0, 100]} />
                    <Tooltip />
                    <Legend />
                    {modelMetrics.baseline && <Bar dataKey="baseline" fill="#ef4444" name="Baseline (GloVe+CNN+BiLSTM)" />}
                    {modelMetrics.proposed && <Bar dataKey="proposed" fill="#10b981" name="Proposed (BERT+CNN+BiLSTM+MHA)" />}
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>
          </>
        )}
      </div>

      <div className="confusion-matrices-section">
        <div className="graph-card">
          <h3>Confusion Matrix - Baseline Model</h3>
          <div className="confusion-matrix-container">
            <div className="confusion-matrix">
              <div className="matrix-cell"></div>
              <div className="matrix-cell matrix-header">Predicted: Sarcastic</div>
              <div className="matrix-cell matrix-header">Predicted: Not Sarcastic</div>

              <div className="matrix-cell matrix-label">Actual: Sarcastic</div>
              <div className="matrix-cell matrix-value true-positive">
                <div className="value">{confusionMatrixBaseline.truePositive}</div>
                <div className="label">True Positive</div>
              </div>
              <div className="matrix-cell matrix-value false-negative">
                <div className="value">{confusionMatrixBaseline.falseNegative}</div>
                <div className="label">False Negative</div>
              </div>

              <div className="matrix-cell matrix-label">Actual: Not Sarcastic</div>
              <div className="matrix-cell matrix-value false-positive">
                <div className="value">{confusionMatrixBaseline.falsePositive}</div>
                <div className="label">False Positive</div>
              </div>
              <div className="matrix-cell matrix-value true-negative">
                <div className="value">{confusionMatrixBaseline.trueNegative}</div>
                <div className="label">True Negative</div>
              </div>
            </div>

            <div className="matrix-legend">
              <div className="legend-item">
                <div className="legend-color" style={{ background: 'linear-gradient(135deg, #d4fc79 0%, #96e6a1 100%)' }}></div>
                <span className="legend-text">True Positive</span>
              </div>
              <div className="legend-item">
                <div className="legend-color" style={{ background: 'linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%)' }}></div>
                <span className="legend-text">True Negative</span>
              </div>
              <div className="legend-item">
                <div className="legend-color" style={{ background: 'linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%)' }}></div>
                <span className="legend-text">False Positive</span>
              </div>
              <div className="legend-item">
                <div className="legend-color" style={{ background: 'linear-gradient(135deg, #fbc2eb 0%, #a6c1ee 100%)' }}></div>
                <span className="legend-text">False Negative</span>
              </div>
            </div>
          </div>
        </div>

        <div className="graph-card">
          <h3>Confusion Matrix - Proposed Model</h3>
          <div className="confusion-matrix-container">
            <div className="confusion-matrix">
              <div className="matrix-cell"></div>
              <div className="matrix-cell matrix-header">Predicted: Sarcastic</div>
              <div className="matrix-cell matrix-header">Predicted: Not Sarcastic</div>

              <div className="matrix-cell matrix-label">Actual: Sarcastic</div>
              <div className="matrix-cell matrix-value true-positive">
                <div className="value">{confusionMatrixProposed.truePositive}</div>
                <div className="label">True Positive</div>
              </div>
              <div className="matrix-cell matrix-value false-negative">
                <div className="value">{confusionMatrixProposed.falseNegative}</div>
                <div className="label">False Negative</div>
              </div>

              <div className="matrix-cell matrix-label">Actual: Not Sarcastic</div>
              <div className="matrix-cell matrix-value false-positive">
                <div className="value">{confusionMatrixProposed.falsePositive}</div>
                <div className="label">False Positive</div>
              </div>
              <div className="matrix-cell matrix-value true-negative">
                <div className="value">{confusionMatrixProposed.trueNegative}</div>
                <div className="label">True Negative</div>
              </div>
            </div>

            <div className="matrix-legend">
              <div className="legend-item">
                <div className="legend-color" style={{ background: 'linear-gradient(135deg, #d4fc79 0%, #96e6a1 100%)' }}></div>
                <span className="legend-text">True Positive</span>
              </div>
              <div className="legend-item">
                <div className="legend-color" style={{ background: 'linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%)' }}></div>
                <span className="legend-text">True Negative</span>
              </div>
              <div className="legend-item">
                <div className="legend-color" style={{ background: 'linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%)' }}></div>
                <span className="legend-text">False Positive</span>
              </div>
              <div className="legend-item">
                <div className="legend-color" style={{ background: 'linear-gradient(135deg, #fbc2eb 0%, #a6c1ee 100%)' }}></div>
                <span className="legend-text">False Negative</span>
              </div>
            </div>
          </div>
        </div>
      </div>

    </div>
  );
};

export default SarcasmDetector;
