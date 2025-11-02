import { createSignal, onMount, createMemo, Show, For } from 'solid-js';
import { loadModel, classifyImage, getModelStatus, isModelReady } from './lib/classify';
import { createConsola } from 'consola/browser';
import type { Component } from 'solid-js';

const log = createConsola({ level: 4 }).withTag('visagist');

interface ClassificationResult {
  id: string;
  filename: string;
  imageUrl: string;
  isMeme: boolean;
  confidence: number;
  inferenceTimeMs: number;
  status: 'loading' | 'completed' | 'error';
  error?: string;
  timestamp: number;
}

const App: Component = () => {
  const [modelReady, setModelReady] = createSignal(false);
  const [modelStatus, setModelStatus] = createSignal('loading');
  const [results, setResults] = createSignal<ClassificationResult[]>([]);
  const [dragOver, setDragOver] = createSignal(false);
  const [totalProcessed, setTotalProcessed] = createSignal(0);

  let fileInputRef: HTMLInputElement;

  // calculate average accuracy for meme detection
  const averageAccuracy = createMemo(() => {
    const completed = results().filter(r => r.status === 'completed');
    if (completed.length === 0) return 0;

    const sum = completed.reduce((acc, r) => acc + r.confidence, 0);
    return Math.round((sum / completed.length) * 100);
  });

  // count memes detected
  const memeCount = createMemo(() => {
    return results().filter(r => r.status === 'completed' && r.isMeme).length;
  });

  onMount(async () => {
    log.info('visagist initializing neural networks');
    await initializeModel();
  });

  const initializeModel = async () => {
    try {
      setModelStatus('downloading');
      log.info('downloading ai model architecture');

      await loadModel();

      setModelReady(true);
      setModelStatus('ready');
      log.info('neural network initialized successfully');
    } catch (error) {
      setModelReady(false);
      setModelStatus('error');
      log.error('failed to initialize neural network', {
        error: error instanceof Error ? error.message : String(error)
      });
    }
  };

  const handleFileSelect = (files: FileList | null) => {
    if (!files || files.length === 0) return;

    Array.from(files).forEach(file => {
      if (file.type.startsWith('image/')) {
        processFile(file);
      }
    });
  };

  const processFile = async (file: File) => {
    const resultId = `${Date.now()}_${Math.random()}`;
    const imageUrl = URL.createObjectURL(file);

    // add loading result with timestamp
    const loadingResult: ClassificationResult = {
      id: resultId,
      filename: file.name,
      imageUrl,
      isMeme: false,
      confidence: 0,
      inferenceTimeMs: 0,
      status: 'loading',
      timestamp: Date.now()
    };

    setResults(prev => [loadingResult, ...prev]);
    setTotalProcessed(p => p + 1);

    if (!isModelReady()) {
      setResults(prev => prev.map(r =>
        r.id === resultId
          ? { ...r, status: 'error' as const, error: 'neural network not initialized' }
          : r
      ));
      return;
    }

    try {
      const startTime = performance.now();
      const result = await classifyImage(imageUrl);
      const processingTime = performance.now() - startTime;

      setResults(prev => prev.map(r =>
        r.id === resultId
          ? {
            ...r,
            isMeme: result.isMeme,
            confidence: result.confidence,
            inferenceTimeMs: processingTime,
            status: 'completed' as const
          }
          : r
      ));

      log.info('classification complete', {
        filename: file.name,
        isMeme: result.isMeme,
        confidence: result.confidence,
        inferenceTimeMs: processingTime
      });

    } catch (error) {
      log.error('classification failed', {
        error: error instanceof Error ? error.message : String(error),
        filename: file.name
      });

      setResults(prev => prev.map(r =>
        r.id === resultId
          ? {
            ...r,
            status: 'error' as const,
            error: error instanceof Error ? error.message : String(error)
          }
          : r
      ));
    }
  };

  const handleDrop = (e: DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    handleFileSelect(e.dataTransfer?.files);
  };

  const handleDragOver = (e: DragEvent) => {
    e.preventDefault();
    setDragOver(true);
  };

  const handleDragLeave = (e: DragEvent) => {
    e.preventDefault();
    setDragOver(false);
  };

  const clearResults = () => {
    // clean up object urls before clearing
    results().forEach(r => URL.revokeObjectURL(r.imageUrl));
    setResults([]);
  };

  const exportResults = () => {
    const data = results()
      .filter(r => r.status === 'completed')
      .map(r => ({
        filename: r.filename,
        isMeme: r.isMeme,
        confidence: r.confidence,
        timestamp: new Date(r.timestamp).toISOString()
      }));

    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `visagist_analysis_${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const getStatusMessage = () => {
    switch (modelStatus()) {
      case 'ready': return 'online';
      case 'downloading': return 'initializing';
      case 'loading': return 'booting';
      case 'error': return 'offline';
      default: return 'unknown';
    }
  };

  const getResultLabel = (result: ClassificationResult) => {
    if (result.status === 'completed') {
      return result.isMeme ? 'certified meme' : 'regular image';
    }
    return '';
  };

  return (
    <div class="app">
      <header class="hero">
        <div class="title-block">
          <h1>visagist</h1>
          <div class="subtitle">
            100% local<span class="highlight">ai meme detection</span>
          </div>
        </div>

        <div class={`status-badge ${modelStatus()}`}>
          {getStatusMessage()}
        </div>
      </header>

      <main class="workspace">
        <div class="upload-area">
          <div
            class={`drop-target ${dragOver() ? 'active' : ''} ${!modelReady() ? 'disabled' : ''}`}
            onDrop={handleDrop}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onClick={() => modelReady() && fileInputRef.click()}
          >
            <Show
              when={modelReady()}
              fallback={
                <div class="loading-text">
                  neural network initializing...
                </div>
              }
            >
              <div class="upload-icon">↑</div>
              <div class="upload-text">
                drop images here
                <span class="format-info">supports jpg, png, webp, gif</span>
              </div>
            </Show>
          </div>

          <input
            ref={fileInputRef!}
            type="file"
            multiple
            accept="image/*"
            style="display: none"
            onChange={(e) => handleFileSelect(e.target.files)}
          />
        </div>

        <Show when={results().length > 0}>
          <div class="controls">
            <div class="button-group">
              <button class="clear-button" onClick={clearResults}>
                clear all
              </button>
              <Show when={results().some(r => r.status === 'completed')}>
                <button class="export-button" onClick={exportResults}>
                  export data
                </button>
              </Show>
            </div>

            <div class="results-info">
              <div class="results-count">
                {results().length} analyzed
              </div>
              <Show when={averageAccuracy() > 0}>
                <div class="accuracy-meter">
                  <span class="accuracy-label">accuracy</span>
                  <span class="accuracy-value">{averageAccuracy()}%</span>
                </div>
              </Show>
            </div>
          </div>
        </Show>

        <div class="results-grid">
          <For each={results()}>
            {(result) => (
              <article class={`result-card ${result.status} ${result.isMeme ? 'positive' : 'negative'}`}>
                <div class="image-container">
                  <img
                    src={result.imageUrl}
                    alt={`analysis of ${result.filename}`}
                    loading="lazy"
                  />
                  <div class="overlay">
                    <Show when={result.status === 'loading'}>
                      <div class="spinner"></div>
                    </Show>
                    <Show when={result.status === 'completed'}>
                      <div class="result-label">
                        {getResultLabel(result)}
                      </div>
                    </Show>
                    <Show when={result.status === 'error'}>
                      <div class="error-icon">×</div>
                    </Show>
                  </div>
                </div>

                <div class="metadata">
                  <div class="filename">{result.filename}</div>
                  <Show when={result.status === 'completed'}>
                    <div class="stats">
                      <span class="stat-badge confidence">
                        {Math.round(result.confidence * 100)}% match
                      </span>
                      <span class="stat-badge timing">
                        {result.inferenceTimeMs.toFixed(0)}ms
                      </span>
                    </div>
                  </Show>
                  <Show when={result.status === 'error'}>
                    <div class="error-message">analysis failed</div>
                  </Show>
                </div>
              </article>
            )}
          </For>
        </div>
      </main>
    </div>
  );
};

export default App;
