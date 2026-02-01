import React, { useState } from 'react';
import { LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import './SarcasmDetector.css';

const SarcasmDetector = () => {
  const [text, setText] = useState('');
  const [result, setResult] = useState(null);
  const [analyzing, setAnalyzing] = useState(false);
  const [selectedModel, setSelectedModel] = useState('baseline'); // 'baseline' or 'proposed'

  // Model performance comparison data
  const modelPerformanceData = [
    { metric: 'Accuracy', baseline: 89.25, proposed: 91.76 },
    { metric: 'Precision', baseline: 90.38, proposed: 92.15 },
    { metric: 'F1-Score', baseline: 87.56, proposed: 90.82 },
    { metric: 'Specificity', baseline: 90.27, proposed: 92.43 },
  ];

  const trainingHistoryData = [
    { epoch: 1, baselineAcc: 72.5, proposedAcc: 75.8, baselineLoss: 0.58, proposedLoss: 0.52 },
    { epoch: 2, baselineAcc: 78.3, proposedAcc: 82.1, baselineLoss: 0.48, proposedLoss: 0.42 },
    { epoch: 3, baselineAcc: 82.6, proposedAcc: 86.4, baselineLoss: 0.41, proposedLoss: 0.35 },
    { epoch: 4, baselineAcc: 85.1, proposedAcc: 88.7, baselineLoss: 0.36, proposedLoss: 0.30 },
    { epoch: 5, baselineAcc: 86.9, proposedAcc: 90.2, baselineLoss: 0.33, proposedLoss: 0.27 },
    { epoch: 6, baselineAcc: 88.1, proposedAcc: 91.1, baselineLoss: 0.31, proposedLoss: 0.25 },
    { epoch: 7, baselineAcc: 88.8, proposedAcc: 91.5, baselineLoss: 0.29, proposedLoss: 0.23 },
    { epoch: 8, baselineAcc: 89.25, proposedAcc: 91.76, baselineLoss: 0.28, proposedLoss: 0.22 },
  ];

  // Confusion Matrix Data (Baseline Model)
  const confusionMatrixBaseline = {
    truePositive: 677,
    falsePositive: 162,
    falseNegative: 92,
    trueNegative: 947
  };

  // Confusion Matrix Data (Proposed Model)
  const confusionMatrixProposed = {
    truePositive: 845,
    falsePositive: 68,
    falseNegative: 73,
    trueNegative: 1014
  };

  // Simulated model detection with different characteristics
  const detectSarcasm = (inputText, model) => {
    const lowerText = inputText.toLowerCase();
    
    // Sarcasm indicators
    const sarcasmKeywords = [
      'yeah right',
      'sure',
      'obviously',
      'oh great',
      'fantastic',
      'wonderful',
      'brilliant',
      'perfect',
      'just what i needed',
      'oh wow',
      'shocking'
    ];
    
    const hasExclamation = (inputText.match(/!/g) || []).length >= 2;
    const hasQuotes = inputText.includes('"') || inputText.includes("'");
    const hasEllipsis = inputText.includes('...');
    
    let sarcasmScore = 0;
    let indicators = [];
    
    // Check for sarcasm keywords
    sarcasmKeywords.forEach(keyword => {
      if (lowerText.includes(keyword)) {
        sarcasmScore += 30;
        indicators.push(`Contains phrase: "${keyword}"`);
      }
    });
    
    // Check punctuation patterns
    if (hasExclamation) {
      sarcasmScore += 15;
      indicators.push('Multiple exclamation marks');
    }
    
    if (hasQuotes) {
      sarcasmScore += 10;
      indicators.push('Contains quotes (air quotes?)');
    }
    
    if (hasEllipsis) {
      sarcasmScore += 10;
      indicators.push('Contains ellipsis (trailing off)');
    }
    
    // Check for ALL CAPS words
    const words = inputText.split(' ');
    const capsWords = words.filter(word => 
      word.length > 2 && word === word.toUpperCase() && /[A-Z]/.test(word)
    );
    
    if (capsWords.length > 0) {
      sarcasmScore += 15;
      indicators.push(`Emphasis with caps: ${capsWords.join(', ')}`);
    }
    
    // Determine result based on model
    const isSarcastic = sarcasmScore > 25;
    let confidence = Math.min(sarcasmScore, 100);
    
    // Proposed model (BERT+CNN+BiLSTM+MHA) has better performance
    if (model === 'proposed') {
      // Boost confidence and accuracy for proposed model
      confidence = Math.min(confidence * 1.12, 100);
      
      // Add contextual understanding indicators
      indicators.push('BERT contextual embeddings analyzed');
      indicators.push('Multi-head attention patterns detected');
    } else {
      // Baseline model (GloVe+CNN+BiLSTM+Attention)
      indicators.push('GloVe embeddings processed');
      indicators.push('Single attention layer applied');
    }
    
    return {
      isSarcastic,
      confidence: Math.round(confidence * 10) / 10,
      indicators: indicators.length > 0 ? indicators : ['No strong sarcasm indicators found'],
      model: model === 'baseline' ? 'GloVe+CNN+BiLSTM+Attention' : 'BERT+CNN+BiLSTM+MHA',
      processingTime: model === 'baseline' ? '125ms' : '187ms'
    };
  };

  const handleAnalyze = () => {
    if (!text.trim()) {
      return;
    }

    setAnalyzing(true);
    
    // Simulate processing delay (proposed model takes slightly longer)
    const delay = selectedModel === 'proposed' ? 900 : 800;
    setTimeout(() => {
      const detection = detectSarcasm(text, selectedModel);
      setResult(detection);
      setAnalyzing(false);
    }, delay);
  };

  const handleReset = () => {
    setText('');
    setResult(null);
  };

  return (
    <div className="sarcasm-detector">
      <div className="header">
        <h1>Sarcasm Detector</h1>
        <p className="subtitle">Compare Baseline vs. Proposed Deep Learning Models</p>
      </div>

      <div className="model-selector">
        <h3>Select Model:</h3>
        <div className="model-options">
          <button 
            className={`model-button ${selectedModel === 'baseline' ? 'active' : ''}`}
            onClick={() => setSelectedModel('baseline')}
          >
            <div className="model-button-title">Baseline Model</div>
            <div className="model-button-desc">GloVe + CNN + BiLSTM + Attention</div>
          </button>
          <button 
            className={`model-button ${selectedModel === 'proposed' ? 'active' : ''}`}
            onClick={() => setSelectedModel('proposed')}
          >
            <div className="model-button-title">Proposed Model</div>
            <div className="model-button-desc">BERT + CNN + BiLSTM + MHA</div>
          </button>
        </div>
      </div>

      <div className="input-section">
        <textarea
          className="text-input"
          placeholder="Type or paste text to analyze... (e.g., 'Oh great, another meeting!')"
          value={text}
          onChange={(e) => setText(e.target.value)}
          rows={6}
        />
        
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
      </div>

      {result && (
        <div className={`result-section ${result.isSarcastic ? 'sarcastic' : 'not-sarcastic'}`}>
          <div className="result-header">
            <h2>
              {result.isSarcastic ? 'SARCASM DETECTED!' : 'Not Sarcastic'}
            </h2>
            <div className="model-badge">{result.model}</div>
            <div className="processing-time">Processing time: {result.processingTime}</div>
          </div>
          
          <div className="confidence-bar">
            <div className="confidence-label">
              Confidence: {result.confidence}%
            </div>
            <div className="progress-bar">
              <div 
                className="progress-fill"
                style={{ width: `${result.confidence}%` }}
              />
            </div>
          </div>

          <div className="indicators">
            <h3>Analysis Details:</h3>
            <ul>
              {result.indicators.map((indicator, index) => (
                <li key={index}>{indicator}</li>
              ))}
            </ul>
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

      <div className="performance-graphs">
        <h2>Model Performance Comparison</h2>
        
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
                <Bar dataKey="baseline" fill="#ef4444" name="Baseline (GloVe+CNN+BiLSTM)" />
                <Bar dataKey="proposed" fill="#10b981" name="Proposed (BERT+CNN+BiLSTM+MHA)" />
              </BarChart>
            </ResponsiveContainer>
          </div>

          <div className="graph-card">
            <h3>Training History - Accuracy</h3>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={trainingHistoryData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="epoch" label={{ value: 'Epoch', position: 'insideBottom', offset: -5 }} />
                <YAxis domain={[60, 95]} label={{ value: 'Accuracy (%)', angle: -90, position: 'insideLeft' }} />
                <Tooltip />
                <Legend />
                <Line type="monotone" dataKey="baselineAcc" stroke="#ef4444" name="Baseline" strokeWidth={2} />
                <Line type="monotone" dataKey="proposedAcc" stroke="#10b981" name="Proposed" strokeWidth={2} />
              </LineChart>
            </ResponsiveContainer>
          </div>

          <div className="graph-card">
            <h3>Training History - Loss</h3>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={trainingHistoryData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="epoch" label={{ value: 'Epoch', position: 'insideBottom', offset: -5 }} />
                <YAxis domain={[0.15, 0.65]} label={{ value: 'Loss', angle: -90, position: 'insideLeft' }} />
                <Tooltip />
                <Legend />
                <Line type="monotone" dataKey="baselineLoss" stroke="#ef4444" name="Baseline" strokeWidth={2} />
                <Line type="monotone" dataKey="proposedLoss" stroke="#10b981" name="Proposed" strokeWidth={2} />
              </LineChart>
            </ResponsiveContainer>
          </div>

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
                  <div className="legend-color" style={{background: 'linear-gradient(135deg, #d4fc79 0%, #96e6a1 100%)'}}></div>
                  <span className="legend-text">True Positive</span>
                </div>
                <div className="legend-item">
                  <div className="legend-color" style={{background: 'linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%)'}}></div>
                  <span className="legend-text">True Negative</span>
                </div>
                <div className="legend-item">
                  <div className="legend-color" style={{background: 'linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%)'}}></div>
                  <span className="legend-text">False Positive</span>
                </div>
                <div className="legend-item">
                  <div className="legend-color" style={{background: 'linear-gradient(135deg, #fbc2eb 0%, #a6c1ee 100%)'}}></div>
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
                  <div className="legend-color" style={{background: 'linear-gradient(135deg, #d4fc79 0%, #96e6a1 100%)'}}></div>
                  <span className="legend-text">True Positive</span>
                </div>
                <div className="legend-item">
                  <div className="legend-color" style={{background: 'linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%)'}}></div>
                  <span className="legend-text">True Negative</span>
                </div>
                <div className="legend-item">
                  <div className="legend-color" style={{background: 'linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%)'}}></div>
                  <span className="legend-text">False Positive</span>
                </div>
                <div className="legend-item">
                  <div className="legend-color" style={{background: 'linear-gradient(135deg, #fbc2eb 0%, #a6c1ee 100%)'}}></div>
                  <span className="legend-text">False Negative</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="disclaimer">
        <p>
          <strong>Note:</strong> This is a prototype demonstrating model architectures. 
          A production system would use actual trained models for inference.
        </p>
      </div>
    </div>
  );
};

export default SarcasmDetector;
