let evidenceLoaded = false;
let evidenceLoading = false;

function evidenceElement(tag, text, className) {
  const element = document.createElement(tag);
  if (text) element.textContent = text;
  if (className) element.className = className;
  return element;
}

function evidenceLink(label, group, filename, download = false) {
  const link = evidenceElement('a', label, 'btn');
  link.href = '/evidence/' + group + '/' + encodeURIComponent(filename);
  if (download) link.download = filename;
  return link;
}

function evidencePanel(title, description) {
  const panel = evidenceElement('article', '', 'panel');
  panel.append(evidenceElement('h2', title));
  if (description) panel.append(evidenceElement('p', description, 'evidence-copy'));
  return panel;
}

function evidenceTable(headers, rows) {
  const wrapper = evidenceElement('div', '', 'twrap');
  const table = evidenceElement('table', '', 'tbl evidence-table');
  const head = table.createTHead().insertRow();
  for (const label of headers) {
    const cell = evidenceElement('th', label);
    cell.scope = 'col';
    head.append(cell);
  }
  const body = table.createTBody();
  for (const [label, ...values] of rows) {
    const row = body.insertRow();
    const heading = evidenceElement('th', String(label));
    heading.scope = 'row';
    row.append(heading);
    for (const value of values) row.insertCell().textContent = value;
  }
  wrapper.append(table);
  return wrapper;
}

function evidenceVideo(title, file) {
  const panel = evidencePanel(title);
  const video = evidenceElement('video');
  video.controls = true;
  video.playsInline = true;
  video.preload = 'none';
  video.setAttribute('aria-label', title);
  const poster = window.WattwiseMedia?.evidencePoster(file);
  if (poster) video.poster = poster;
  video.src = '/evidence/videos/' + encodeURIComponent(file);
  panel.append(video, evidenceLink('Download MP4', 'videos', file, true));
  return panel;
}

function renderSimulationEvidence(data) {
  const value = (number, digits = 2) => number === '' || number == null ? 'N/A' : Number(number).toFixed(digits);
  const percent = fraction => value(Number(fraction) * 100, 1) + '%';
  const names = {
    fixed_original: 'Conventional fixed-speed IK',
    fixed_slow: 'Fixed-slow IK',
    proportional: 'Proportional-speed IK',
    adaptive: 'Adaptive-speed IK',
    teacher_original: 'Original teacher DQN',
    teacher_best: 'Selected teacher DQN',
    teacher_finetuned: 'Fine-tuned teacher DQN',
    student_original: 'Original student DQN',
    student_best: 'Improved lightweight student DQN',
  };
  const comparison = evidenceVideo('Combined 2×2 comparison', data.comparison);
  comparison.querySelector('h2').after(evidenceElement('p',
    'Four separately initialized runs on seed ' + data.seed + ', with a ' + data.max_steps +
    '-step limit per run. Energy and power are simulation estimates.', 'evidence-copy'));
  const results = data.recordings.map(result => [
    result.label, result.success ? 'Completed' : 'Incomplete', result.steps,
    value(result.elapsed_s, 4), value(result.distance, 4), value(result.energy_j),
    value(result.mechanical_j), value(result.peak_w), value(result.latency_ms, 4),
    result.parameters.toLocaleString('en-US'),
  ]);
  const measurementHeaders = [
    'Controller', 'Outcome', 'Steps', 'Simulated time (s)', 'Final error (m)', 'Grid energy (J)',
    'Mechanical energy (J)', 'Peak power (W)', 'Last-step decision (ms)', 'Parameters',
  ];
  comparison.append(evidenceTable(measurementHeaders, results));

  const recordings = evidenceElement('div', '', 'evidence-grid');
  for (const [index, result] of data.recordings.entries()) {
    const panel = evidenceVideo(result.label, result.file);
    const wrapper = evidenceTable(['Measurement', 'Value'],
      measurementHeaders.slice(1).map((label, column) => [label, results[index][column + 1]]));
    wrapper.classList.add('evidence-measurements');
    panel.append(wrapper);
    recordings.append(panel);
  }

  const policies = evidencePanel('Held-out policy results',
    'Nine policies evaluated on the same 50 seeds. Total energy per completion includes the energy spent on failed attempts.');
  policies.append(evidenceTable(
    ['Policy', 'Completed', 'Wilson 95% CI', 'Mean grid J / attempt', 'Mean grid J / success',
      'Mean grid J / failure', 'Total grid J / completion', 'Mean peak W'],
    data.policies.map(row => [names[row.policy], row.completions + '/' + row.episodes + ' (' +
      percent(row.completion_rate) + ')', percent(row.completion_wilson95_low) + ' to ' +
      percent(row.completion_wilson95_high), value(row.mean_grid_energy_all_attempts_j),
      value(row.mean_grid_energy_success_j), value(row.mean_grid_energy_failed_j),
      value(row.aggregate_energy_per_completed_task_j), value(row.mean_peak_power_w)])
  ));

  const paired = evidencePanel('Paired energy vs conventional IK',
    'Energy savings use seeds where both controllers completed. Negative values indicate higher simulated grid energy than conventional IK. N/A indicates zero shared successes.');
  paired.append(evidenceTable(
    ['Policy', 'Shared successes', 'Mean saving (%)', 'Median saving (%)', 'Bootstrap 95% CI (%)'],
    data.paired.filter(row => row.policy !== 'fixed_original').map(row => [
      names[row.policy], row.shared_success_scenarios, value(row.mean_paired_energy_saving_pct),
      value(row.median_paired_energy_saving_pct),
      row.bootstrap_ci95_low_pct === '' ? 'N/A' :
        value(row.bootstrap_ci95_low_pct) + ' to ' + value(row.bootstrap_ci95_high_pct),
    ])
  ));

  const benchmark = evidencePanel('Model size and benchmark',
    'Recorded batch-1 neural-network forward passes on Colab x86_64 CUDA. The video latency column shows last-step controller decisions for the recorded runs.');
  benchmark.append(evidenceTable(
    ['Model', 'Parameters', 'State bytes', 'Passes', 'Median forward pass (ms)', 'Parameter reduction (%)'],
    data.benchmark.map(row => [row.model, row.parameters, row.size_bytes, row.passes,
      value(row.median_ms, 3), value(row.parameters_reduction_vs_teacher_pct)])
  ));

  const graphs = evidencePanel('Research graphs');
  const gallery = evidenceElement('div', '', 'evidence-grid');
  for (const figure of data.figures) {
    const item = evidenceElement('div');
    const link = evidenceLink(figure.label, 'figures', figure.file);
    link.classList.remove('btn');
    const image = evidenceElement('img');
    image.src = link.href;
    image.alt = figure.label;
    image.loading = 'lazy';
    link.replaceChildren(image);
    item.append(evidenceElement('h3', figure.label), link);
    gallery.append(item);
  }
  graphs.append(gallery);

  const downloads = evidencePanel('Download evidence');
  const links = evidenceElement('div', '', 'evidence-links');
  for (const file of data.downloads) {
    links.append(evidenceLink(file, 'downloads', file, true));
  }
  downloads.append(links);
  return [comparison, recordings, policies, paired, benchmark, graphs, downloads];
}

async function loadSimulationEvidence() {
  if (evidenceLoaded || evidenceLoading) return;
  const status = document.getElementById('evidence-status');
  evidenceLoading = true;
  status.hidden = false;
  status.textContent = 'Loading local research evidence…';
  try {
    const response = await fetch('/api/simulation-evidence');
    if (!response.ok) throw new Error('Evidence request returned ' + response.status);
    const panels = renderSimulationEvidence(await response.json());
    const content = document.getElementById('evidence-content');
    content.replaceChildren(...panels);
    const videos = [...content.querySelectorAll('video')];
    for (const video of videos) {
      video.addEventListener('play', () => {
        for (const other of videos) {
          if (other !== video && !other.paused) other.pause();
        }
      });
    }
    evidenceLoaded = true;
    status.hidden = true;
  } catch (error) {
    console.error(error);
    const retry = evidenceElement('button', 'Retry', 'btn');
    retry.type = 'button';
    retry.addEventListener('click', loadSimulationEvidence);
    status.replaceChildren('The local research evidence could not be loaded. ', retry);
  } finally {
    evidenceLoading = false;
  }
}
