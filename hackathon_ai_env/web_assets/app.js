const state = {
  latestEval: [],
  latestInference: null,
  latestStatus: null,
  trainingTimeline: [],
};

const elements = {
  answerExplanation: document.getElementById("answer-explanation"),
  answerMeta: document.getElementById("answer-meta"),
  answerDetail: document.getElementById("answer-detail"),
  answerOutput: document.getElementById("answer-output"),
  feedbackAppliedContainer: document.getElementById("feedback-applied-container"),
  feedbackAppliedText: document.getElementById("feedback-applied-text"),
  answerSummary: document.getElementById("answer-summary"),
  episodesInput: document.getElementById("episodes-input"),
  epsilonIndicator: document.getElementById("epsilon-indicator"),
  evalCount: document.getElementById("eval-count"),
  evalForm: document.getElementById("eval-form"),
  evaluationDetail: document.getElementById("evaluation-detail"),
  evaluationSummary: document.getElementById("evaluation-summary"),
  evaluationTable: document.getElementById("evaluation-table"),
  feedbackCard: document.getElementById("feedback-card"),
  feedbackForm: document.getElementById("feedback-form"),
  feedbackLoopDetail: document.getElementById("feedback-loop-detail"),
  feedbackLoopSummary: document.getElementById("feedback-loop-summary"),
  feedbackNotes: document.getElementById("feedback-notes"),
  feedbackRating: document.getElementById("feedback-rating"),
  feedbackResult: document.getElementById("feedback-result"),
  feedbackStatus: document.getElementById("feedback-status"),
  feedbackSubmit: document.getElementById("feedback-submit"),
  graphStatus: document.getElementById("graph-status"),
  memoryCount: document.getElementById("memory-count"),
  memoryBankCount: document.getElementById("memory-bank-count"),
  memoryBankPanel: document.getElementById("memory-bank-panel"),
  memoryTopWeight: document.getElementById("memory-top-weight"),
  persistenceDetail: document.getElementById("persistence-detail"),
  persistenceSummary: document.getElementById("persistence-summary"),
  notice: document.getElementById("notice"),
  qStateCount: document.getElementById("q-state-count"),
  qTableCount: document.getElementById("q-table-count"),
  qTablePanel: document.getElementById("q-table-panel"),
  queryHelperCopy: document.getElementById("query-helper-copy"),
  queryInput: document.getElementById("query-input"),
  queryModePill: document.getElementById("query-mode-pill"),
  resetButton: document.getElementById("reset-button"),
  resultRoute: document.getElementById("result-route"),
  scenarioCount: document.getElementById("scenario-count"),
  sessionFeedback: document.getElementById("session-feedback"),
  sessionFeedbackDetail: document.getElementById("session-feedback-detail"),
  sessionHealth: document.getElementById("session-health"),
  sessionHealthDetail: document.getElementById("session-health-detail"),
  sessionPersistence: document.getElementById("session-persistence"),
  sessionPersistenceDetail: document.getElementById("session-persistence-detail"),
  stepGraph: document.getElementById("step-graph"),
  storagePathDetail: document.getElementById("storage-path-detail"),
  storagePathLabel: document.getElementById("storage-path-label"),
  timeline: document.getElementById("timeline"),
  timelineRange: document.getElementById("timeline-range"),
  trainedIndicator: document.getElementById("trained-indicator"),
  trainForm: document.getElementById("train-form"),
  trainingDetail: document.getElementById("training-detail"),
  trainingSummary: document.getElementById("training-summary"),
  warmMemoryInput: document.getElementById("warm-memory-input"),
  askForm: document.getElementById("ask-form"),
};

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.error || "Request failed");
  }
  return payload;
}

function setNotice(message, tone = "neutral") {
  elements.notice.textContent = message;
  elements.notice.className = `notice ${tone}`;
}

function syncNoticeWithStatus(status) {
  const persistence = status.persistence || { enabled: false };
  const memoryCount = Number(status.memory_snapshot?.count || 0);

  if (status.feedback_pending) {
    setNotice("The latest answer is waiting for your rating so the reward can update the policy.", "neutral");
    return;
  }

  if (status.last_feedback) {
    setNotice(
      persistence.enabled
        ? "Restored a saved session. The latest rated answer and learned state are loaded."
        : "The latest answer has already been rated and written back into the policy.",
      "success"
    );
    return;
  }

  if (status.trained || status.q_state_count > 0 || memoryCount > 0) {
    setNotice(
      persistence.enabled
        ? "Restored a saved session. You can keep training, ask a query, or reset the environment."
        : "The policy is ready. Train again, ask a query, or run evaluation.",
      "success"
    );
    return;
  }

  setNotice("Waiting for your first action.", "neutral");
}

function setBusy(button, busyLabel) {
  const originalLabel = button.textContent;
  button.disabled = true;
  button.textContent = busyLabel;
  return () => {
    button.disabled = false;
    button.textContent = originalLabel;
  };
}

function formatNumber(value) {
  if (typeof value !== "number" || Number.isNaN(value)) {
    return "--";
  }
  return value.toFixed(2);
}

function formatPercent(value) {
  if (typeof value !== "number" || Number.isNaN(value)) {
    return "--";
  }
  return `${formatNumber(value * 100)}%`;
}

function clampValue(value, minimum = 0, maximum = 1) {
  return Math.min(maximum, Math.max(minimum, value));
}

function compactPath(path) {
  if (!path) {
    return "Not configured";
  }
  const value = String(path);
  if (value.length <= 42) {
    return value;
  }
  return `...${value.slice(-39)}`;
}

function readDifficulty(stateKey) {
  const match = stateKey && stateKey.match(/diff=([a-z]+)/);
  return match ? match[1] : "unknown";
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function updateQueryComposerState() {
  if (!elements.queryModePill) {
    return;
  }
  if (elements.warmMemoryInput.checked) {
    elements.queryModePill.textContent = "Memory-assisted query";
    elements.queryHelperCopy.textContent = "The app will warm the benchmark memory bank first, then answer using the current policy and any saved user memory.";
    return;
  }
  elements.queryModePill.textContent = "Fresh query";
  elements.queryHelperCopy.textContent = "The app will skip benchmark warm-up and answer directly from the current policy plus any already-saved user memory.";
}

function renderAnswerMeta(result) {
  if (!result) {
    elements.answerMeta.className = "answer-meta empty";
    elements.answerMeta.textContent = "Response metadata appears here after a custom query.";
    elements.answerExplanation.className = "answer-explanation empty";
    elements.answerExplanation.textContent = "A plain-language explanation of the route will appear here after a query.";
    return;
  }

  const difficulty = readDifficulty(result.state_key).toUpperCase();
  const recallCount = result.recalled_memory_keys?.length || 0;
  elements.answerMeta.className = "answer-meta";
  elements.answerMeta.innerHTML = `
    <span class="answer-chip">Final <strong>${escapeHtml(result.final_agent)}</strong></span>
    <span class="answer-chip">Selected <strong>${escapeHtml(result.selected_agent)}</strong></span>
    <span class="answer-chip">Memory <strong>${result.use_memory ? "on" : "off"}</strong></span>
    <span class="answer-chip">Recalls <strong>${recallCount}</strong></span>
    <span class="answer-chip">Difficulty <strong>${difficulty}</strong></span>
    <span class="answer-chip">Threshold <strong>${formatNumber(result.threshold)}</strong></span>
  `;
  elements.answerExplanation.className = "answer-explanation";
  elements.answerExplanation.textContent = result.explanation || "No explanation available for this answer.";
}

function renderSessionSummary(status) {
  const persistence = status.persistence || { enabled: false, path: null };
  const trained = Boolean(status.trained);
  const memoryCount = Number(status.memory_snapshot?.count || 0);
  const feedbackPending = Boolean(status.feedback_pending);
  const lastFeedback = status.last_feedback || null;

  elements.persistenceSummary.textContent = persistence.enabled ? "Autosave enabled" : "Temporary session";
  elements.persistenceDetail.textContent = persistence.enabled
    ? "The dashboard is saving Q-learning and memory state so it can be restored on restart."
    : "No storage path is active, so a restart will start from a clean session.";
  elements.storagePathLabel.textContent = compactPath(persistence.path);
  elements.storagePathDetail.textContent = persistence.enabled
    ? "For Hugging Face, this path should live on `/data` with persistent storage enabled."
    : "Set `APP_STATE_PATH` if you want a stable save file outside the running process.";

  elements.sessionHealth.textContent = trained ? "Policy ready" : "Needs training";
  elements.sessionHealthDetail.textContent = trained
    ? `${status.q_state_count} Q states and ${memoryCount} memory items are loaded in this session.`
    : "Train once to build the Q-table and populate the dashboard with learned behavior.";

  elements.sessionPersistence.textContent = persistence.enabled ? "Restart-safe session" : "Ephemeral session";
  elements.sessionPersistenceDetail.textContent = persistence.enabled
    ? `State file: ${compactPath(persistence.path)}`
    : "A browser refresh is fine, but a server restart will reset learned state.";

  if (feedbackPending) {
    elements.feedbackLoopSummary.textContent = "Waiting for rating";
    elements.feedbackLoopDetail.textContent = "The latest answer is ready to be rated so the reward can update the policy.";
    elements.sessionFeedback.textContent = "Feedback pending";
    elements.sessionFeedbackDetail.textContent = "Rate the latest answer to complete the RL loop for this turn.";
    return;
  }

  if (lastFeedback) {
    elements.feedbackLoopSummary.textContent = `Last reward ${formatNumber(lastFeedback.reward.total)}`;
    elements.feedbackLoopDetail.textContent = "User feedback has already been written back into the policy and memory.";
    elements.sessionFeedback.textContent = `Rated ${lastFeedback.rating}/5`;
    elements.sessionFeedbackDetail.textContent = lastFeedback.notes
      ? lastFeedback.notes
      : "The last answer was rated and fed back into the system.";
    return;
  }

  elements.feedbackLoopSummary.textContent = "Idle";
  elements.feedbackLoopDetail.textContent = "Ask a query, then rate the answer to show the reinforcement-learning feedback loop.";
  elements.sessionFeedback.textContent = "Idle";
  elements.sessionFeedbackDetail.textContent = "No pending answer to rate yet.";
}

function renderLatestAnswer(result) {
  if (!result) {
    elements.answerSummary.textContent = "No query yet";
    elements.answerDetail.textContent = "Ask a custom question to see agent routing and memory usage.";
    elements.resultRoute.textContent = "Route unavailable";
    elements.answerOutput.textContent = "No custom query has been answered yet.";
    renderAnswerMeta(null);
    if (elements.feedbackAppliedContainer) elements.feedbackAppliedContainer.style.display = "none";
    return;
  }

  const difficulty = readDifficulty(result.state_key);

  elements.answerSummary.textContent = `${result.final_agent} answered (Task: ${difficulty.toUpperCase()})`;
  elements.answerDetail.textContent = `Selected ${result.selected_agent}, memory ${result.use_memory ? "on" : "off"}, threshold ${formatNumber(result.threshold)}, and ${result.recalled_memory_keys?.length || 0} recalled items.`;
  elements.resultRoute.textContent = `State route: ${result.state_key}`;
  renderAnswerMeta(result);
  
  if (result.answer && result.answer.includes("Feedback applied:")) {
      const parts = result.answer.split("Feedback applied:");
      elements.answerOutput.textContent = parts[0].trim();
      elements.feedbackAppliedText.textContent = parts[1].trim();
      if (elements.feedbackAppliedContainer) elements.feedbackAppliedContainer.style.display = "block";
  } else {
      elements.answerOutput.textContent = result.answer;
      if (elements.feedbackAppliedContainer) elements.feedbackAppliedContainer.style.display = "none";
  }
}

function renderStatus(status) {
  state.latestStatus = status;
  elements.trainedIndicator.textContent = status.trained
    ? `Trained for ${status.trained_episodes} episodes`
    : "Not trained";
  elements.epsilonIndicator.textContent = `epsilon ${formatNumber(status.epsilon)}`;
  elements.qStateCount.textContent = status.q_state_count;
  elements.memoryCount.textContent = status.memory_snapshot.count;
  elements.memoryTopWeight.textContent = formatNumber(status.memory_snapshot.top_weight);
  elements.scenarioCount.textContent = status.scenario_count;
  renderPolicy(status.policy || { q_table: [], memory_bank: [] });
  renderFeedbackStatus(status);
  renderSessionSummary(status);
  syncNoticeWithStatus(status);
  state.latestInference = status.last_inference || null;
  state.trainingTimeline = status.training_timeline || [];
  renderLatestAnswer(state.latestInference);
  renderStepGraph(state.latestInference, status.last_feedback);
  renderTimeline(state.trainingTimeline);

  if (status.last_train) {
    elements.trainingSummary.textContent = `Accuracy ${formatNumber(status.last_train.accuracy * 100)}%`;
    elements.trainingDetail.textContent = `Average reward ${formatNumber(status.last_train.average_reward)} with ${status.last_train.memory_items} memory items.`;
  } else {
    elements.trainingSummary.textContent = "No run yet";
    elements.trainingDetail.textContent = "Train the environment to generate a fresh policy summary.";
  }
}

function renderFeedbackStatus(status) {
  const pending = Boolean(status.feedback_pending);
  elements.feedbackRating.disabled = !pending;
  elements.feedbackNotes.disabled = !pending;
  elements.feedbackSubmit.disabled = !pending;

  if (pending) {
    elements.feedbackStatus.textContent = "Waiting for your rating";
    if (!status.last_feedback) {
      elements.feedbackResult.className = "feedback-result pending";
      elements.feedbackResult.textContent = "Rate the latest answer to calculate reward and update the policy.";
    }
    return;
  }

  if (status.last_feedback) {
    const reward = status.last_feedback.reward;
    elements.feedbackStatus.textContent = `Last rating: ${status.last_feedback.rating}/5`;
    elements.feedbackResult.className = "feedback-result";
    elements.feedbackResult.innerHTML = `
      <strong>Reward ${formatNumber(reward.total)}</strong>
      <span>User feedback ${formatNumber(reward.user_feedback)}</span>
      <span>Memory signal ${formatNumber(reward.memory_signal)}</span>
      <span>Confidence alignment ${formatNumber(reward.confidence_alignment)}</span>
      <span>Difficulty bonus ${formatNumber(reward.difficulty_bonus || 0)}</span>
      ${status.last_feedback.notes ? `<p>${escapeHtml(status.last_feedback.notes)}</p>` : ""}
    `;
    return;
  }

  elements.feedbackStatus.textContent = "Ask a query first";
  elements.feedbackResult.className = "feedback-result empty";
  elements.feedbackResult.textContent = "Reward for custom queries will be calculated after you rate the answer.";
}

function renderPolicy(policy) {
  const qTable = policy.q_table || [];
  const memoryBank = policy.memory_bank || [];

  elements.qTableCount.textContent = `${qTable.length} states`;
  if (qTable.length === 0) {
    elements.qTablePanel.className = "stack-list empty";
    elements.qTablePanel.textContent = "Train the environment to inspect learned state-action values.";
  } else {
    elements.qTablePanel.className = "stack-list";
    elements.qTablePanel.innerHTML = qTable
      .map(
        (entry) => `
          <article class="stack-card">
            <div class="stack-head">
              <strong>${escapeHtml(entry.state_key)}</strong>
              <span>best ${formatNumber(entry.best_value)}</span>
            </div>
            <div class="pill-row">
              ${entry.actions
                .map(
                  (action) => `
                    <span class="data-pill">
                      ${escapeHtml(action.action_key)}
                      <b>${formatNumber(action.value)}</b>
                    </span>
                  `
                )
                .join("")}
            </div>
          </article>
        `
      )
      .join("");
  }

  elements.memoryBankCount.textContent = `${memoryBank.length} records`;
  if (memoryBank.length === 0) {
    elements.memoryBankPanel.className = "stack-list empty";
    elements.memoryBankPanel.textContent = "Memory records will appear here after training or evaluation.";
  } else {
    elements.memoryBankPanel.className = "stack-list";
    elements.memoryBankPanel.innerHTML = memoryBank
      .map(
        (entry) => `
          <article class="stack-card">
            <div class="stack-head">
              <strong>${escapeHtml(entry.domain)}</strong>
              <span>weight ${formatNumber(entry.weight)}</span>
            </div>
            <p class="stack-summary">${escapeHtml(entry.summary)}</p>
            <div class="pill-row">
              ${entry.keywords
                .map((keyword) => `<span class="data-pill">${escapeHtml(keyword)}</span>`)
                .join("")}
            </div>
            <div class="stack-meta">
              <span>${escapeHtml(entry.key)}</span>
              <span>accesses ${entry.accesses}</span>
            </div>
          </article>
        `
      )
      .join("");
  }
}

function renderTimeline(timeline) {
  if (!timeline || timeline.length === 0) {
    elements.timeline.className = "timeline empty";
    elements.timeline.textContent = "No training history yet.";
    elements.timelineRange.textContent = "No training yet";
    return;
  }

  const chartMax = Math.max(
    1,
    ...timeline.flatMap((item) => [item.average_reward, item.accuracy])
  );
  const width = 720;
  const height = 260;
  const plotLeft = 58;
  const plotRight = width - 24;
  const plotTop = 18;
  const plotBottom = 208;
  const plotWidth = plotRight - plotLeft;
  const plotHeight = plotBottom - plotTop;
  const latest = timeline[timeline.length - 1];
  const first = timeline[0];
  const bestReward = timeline.reduce(
    (best, item) => (item.average_reward > best.average_reward ? item : best),
    timeline[0]
  );
  const bestAccuracy = timeline.reduce(
    (best, item) => (item.accuracy > best.accuracy ? item : best),
    timeline[0]
  );
  const rewardDelta = latest.average_reward - first.average_reward;
  const rangeLabel = timeline.length === 1
    ? `Episode ${timeline[0].episode}`
    : `Episodes ${timeline[0].episode}-${timeline[timeline.length - 1].episode}`;
  const tickFractions = [1, 0.75, 0.5, 0.25, 0];
  const labelEvery = timeline.length > 10 ? 2 : 1;

  const points = timeline.map((item, index) => {
    const progress = timeline.length === 1 ? 0.5 : index / (timeline.length - 1);
    const x = plotLeft + (progress * plotWidth);
    const rewardValue = clampValue(item.average_reward / chartMax, 0, 1);
    const accuracyValue = clampValue(item.accuracy / chartMax, 0, 1);
    return {
      episode: item.episode,
      reward: item.average_reward,
      accuracy: item.accuracy,
      memoryItems: item.memory_items,
      x,
      rewardY: plotBottom - (rewardValue * plotHeight),
      accuracyY: plotBottom - (accuracyValue * plotHeight),
    };
  });

  const rewardPolyline = points.map((point) => `${point.x},${point.rewardY}`).join(" ");
  const accuracyPolyline = points.map((point) => `${point.x},${point.accuracyY}`).join(" ");
  const rewardArea = [
    `${points[0].x},${plotBottom}`,
    ...points.map((point) => `${point.x},${point.rewardY}`),
    `${points[points.length - 1].x},${plotBottom}`,
  ].join(" ");

  elements.timeline.className = "timeline";
  elements.timelineRange.textContent = `${rangeLabel} · reward + accuracy`;
  elements.timeline.innerHTML = `
    <div class="timeline-shell">
      <div class="timeline-summary-grid">
        <article class="timeline-stat-card">
          <span class="timeline-stat-label">Latest reward</span>
          <strong>${formatNumber(latest.average_reward)}</strong>
          <p>Episode ${latest.episode} with ${latest.memory_items} memory items.</p>
        </article>
        <article class="timeline-stat-card">
          <span class="timeline-stat-label">Latest accuracy</span>
          <strong>${formatPercent(latest.accuracy)}</strong>
          <p>Current training snapshot from the last episode.</p>
        </article>
        <article class="timeline-stat-card">
          <span class="timeline-stat-label">Best reward</span>
          <strong>${formatNumber(bestReward.average_reward)}</strong>
          <p>Episode ${bestReward.episode} reached the highest average reward.</p>
        </article>
        <article class="timeline-stat-card">
          <span class="timeline-stat-label">Peak accuracy</span>
          <strong>${formatPercent(bestAccuracy.accuracy)}</strong>
          <p>Episode ${bestAccuracy.episode}. Reward change is ${rewardDelta >= 0 ? "+" : ""}${formatNumber(rewardDelta)} since episode ${first.episode}.</p>
        </article>
      </div>
      <div class="timeline-chart-card">
        <div class="timeline-legend">
          <span class="timeline-pill reward"><i></i>Average reward</span>
          <span class="timeline-pill accuracy"><i></i>Accuracy</span>
          <span class="timeline-caption">Hover points for episode details</span>
        </div>
        <svg class="timeline-chart" viewBox="0 0 ${width} ${height}" preserveAspectRatio="none" aria-label="Training reward and accuracy chart">
          ${tickFractions
            .map((fraction) => {
              const value = chartMax * fraction;
              const y = plotBottom - (fraction * plotHeight);
              return `
                <line class="timeline-grid-line" x1="${plotLeft}" y1="${y}" x2="${plotRight}" y2="${y}"></line>
                <text class="timeline-axis-label" x="${plotLeft - 10}" y="${y + 4}" text-anchor="end">${escapeHtml(formatPercent(value))}</text>
              `;
            })
            .join("")}
          ${points
            .map((point, index) => {
              if (index % labelEvery !== 0 && index !== points.length - 1) {
                return "";
              }
              return `
                <text class="timeline-axis-label timeline-axis-label-x" x="${point.x}" y="${plotBottom + 24}" text-anchor="middle">E${point.episode}</text>
              `;
            })
            .join("")}
          <polygon class="timeline-area" points="${rewardArea}"></polygon>
          <polyline class="timeline-line reward" points="${rewardPolyline}"></polyline>
          <polyline class="timeline-line accuracy" points="${accuracyPolyline}"></polyline>
          ${points
            .map(
              (point) => `
                <circle class="timeline-point reward" cx="${point.x}" cy="${point.rewardY}" r="5">
                  <title>Episode ${point.episode}: reward ${formatNumber(point.reward)}, accuracy ${formatPercent(point.accuracy)}, memory items ${point.memoryItems}</title>
                </circle>
                <circle class="timeline-point accuracy" cx="${point.x}" cy="${point.accuracyY}" r="4">
                  <title>Episode ${point.episode}: accuracy ${formatPercent(point.accuracy)}, reward ${formatNumber(point.reward)}</title>
                </circle>
              `
            )
            .join("")}
        </svg>
      </div>
    </div>
  `;
}

function renderEvaluation(results) {
  state.latestEval = results || [];
  elements.evalCount.textContent = `${state.latestEval.length} scenarios`;

  if (state.latestEval.length === 0) {
    elements.evaluationTable.className = "evaluation-table empty";
    elements.evaluationTable.textContent = "Evaluation results will appear here.";
    return;
  }

  elements.evaluationTable.className = "evaluation-table";
  elements.evaluationTable.innerHTML = `
    <div class="table-head">
      <span>Query</span>
      <span>Expected</span>
      <span>Chosen</span>
      <span>Reward</span>
    </div>
    ${state.latestEval
      .map(
        (result) => `
          <div class="table-row">
            <span class="query-cell" title="${escapeHtml(result.query)}">${escapeHtml(result.query)}</span>
            <span>${escapeHtml(result.expected_domain)}</span>
            <span>${escapeHtml(result.final_agent)}</span>
            <span>${formatNumber(result.reward.total)}</span>
          </div>
        `
      )
      .join("")}
  `;
}

function renderAskResult(result) {
  state.latestInference = result;
  renderLatestAnswer(result);
  elements.feedbackNotes.value = "";
  elements.feedbackRating.value = "5";
}

function renderStepGraph(result, feedback) {
  if (!result) {
    elements.graphStatus.textContent = "No query yet";
    elements.stepGraph.className = "step-graph empty";
    elements.stepGraph.textContent = "Ask a custom query to visualize each step of the pipeline.";
    return;
  }

  const diffMatch = result.state_key && result.state_key.match(/diff=([a-z]+)/);
  const difficultyLabel = diffMatch ? diffMatch[1].toUpperCase() : "N/A";

  const scores = Object.entries(result.agent_scores || {}).sort((left, right) => right[1] - left[1]);
  const maxScore = Math.max(...scores.map((item) => item[1]), 1);
  const reward = feedback?.reward || null;
  const rewardBars = reward
    ? [
        ["Total", reward.total],
        ["User", reward.user_feedback],
        ["Memory", reward.memory_signal],
        ["Align", reward.confidence_alignment],
      ]
    : [];

  elements.graphStatus.textContent = result.final_agent === result.selected_agent
    ? "Policy followed its first choice"
    : "Fusion overrode the first choice";
  elements.stepGraph.className = "step-graph";
  elements.stepGraph.innerHTML = `
    <div class="graph-flow">
      <article class="flow-card">
        <span class="flow-label">1. Query signal &bull; <strong>${difficultyLabel}</strong> Task</span>
        <strong>${escapeHtml((result.inferred_keywords || []).join(", ") || "general")}</strong>
        <p>State: ${escapeHtml(result.state_key)}</p>
      </article>
      <article class="flow-card">
        <span class="flow-label">2. Policy choice</span>
        <strong>${escapeHtml(result.selected_agent)}</strong>
        <p>Memory ${result.use_memory ? "on" : "off"} | threshold ${formatNumber(result.threshold)}</p>
      </article>
      <article class="flow-card">
        <span class="flow-label">3. Final answer</span>
        <strong>${escapeHtml(result.final_agent)}</strong>
        <p>${result.recalled_memory_keys?.length ? `${result.recalled_memory_keys.length} memory hits` : "fresh answer path"}</p>
      </article>
      <article class="flow-card">
        <span class="flow-label">4. Feedback reward</span>
        <strong>${reward ? formatNumber(reward.total) : "pending"}</strong>
        <p>${reward ? "Updated after your rating" : "Rate the answer to update reward"}</p>
      </article>
    </div>
    <div class="graph-grid">
      <section class="graph-card">
        <div class="graph-card-head">
          <h4>Agent score bars</h4>
          <span>${scores.length} agents</span>
        </div>
        <div class="score-stack">
          ${scores
            .map(
              ([agent, value]) => `
                <div class="score-row">
                  <div class="score-top">
                    <span>${escapeHtml(agent)}</span>
                    <span>${formatNumber(value)}</span>
                  </div>
                  <div class="score-track">
                    <div class="score-fill" style="width:${Math.max(10, (value / maxScore) * 100)}%"></div>
                  </div>
                </div>
              `
            )
            .join("")}
        </div>
      </section>
      <section class="graph-card">
        <div class="graph-card-head">
          <h4>Memory + reward</h4>
          <span>${result.recalled_memory_keys?.length || 0} recalls</span>
        </div>
        <div class="pill-row graph-pills">
          ${(result.recalled_memory_keys?.length
            ? result.recalled_memory_keys
                .map((key) => `<span class="data-pill">${escapeHtml(key)}</span>`)
                .join("")
            : '<span class="graph-empty-copy">No memory keys were used for this answer.</span>')}
        </div>
        <div class="reward-stack ${reward ? "" : "empty"}">
          ${reward
            ? rewardBars
                .map(
                  ([label, value]) => `
                    <div class="score-row">
                      <div class="score-top">
                        <span>${label}</span>
                        <span>${formatNumber(value)}</span>
                      </div>
                      <div class="score-track reward-track">
                        <div class="score-fill reward-fill" style="width:${Math.max(8, value * 100)}%"></div>
                      </div>
                    </div>
                  `
                )
                .join("")
            : '<div class="graph-empty-copy">Submit feedback to see reward bars for this answer.</div>'}
        </div>
      </section>
    </div>
  `;
}

function guideToFeedback() {
  elements.feedbackCard.classList.add("feedback-ready");
  elements.feedbackCard.scrollIntoView({ behavior: "smooth", block: "center" });
  window.setTimeout(() => {
    elements.feedbackRating.focus();
  }, 180);
  window.setTimeout(() => {
    elements.feedbackCard.classList.remove("feedback-ready");
  }, 2200);
}

async function refreshStatus() {
  const payload = await api("/api/status");
  renderStatus(payload.status);
}

elements.trainForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const button = elements.trainForm.querySelector("button");
  const done = setBusy(button, "Training...");
  try {
    const payload = await api("/api/train", {
      method: "POST",
      body: JSON.stringify({ episodes: Number(elements.episodesInput.value) || 40 }),
    });
    renderStatus(payload.status);
    setNotice(payload.message, "success");
  } catch (error) {
    setNotice(error.message, "error");
  } finally {
    done();
  }
});

elements.evalForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const button = elements.evalForm.querySelector("button");
  const done = setBusy(button, "Evaluating...");
  try {
    const payload = await api("/api/eval", {
      method: "POST",
      body: JSON.stringify({ episodes: Number(elements.episodesInput.value) || 40 }),
    });
    renderStatus(payload.status);
    renderEvaluation(payload.results);
    elements.evaluationSummary.textContent = `Accuracy ${formatNumber(payload.summary.accuracy * 100)}%`;
    elements.evaluationDetail.textContent = `Average reward ${formatNumber(payload.summary.average_reward)} across ${payload.summary.results} scenarios.`;
    setNotice(payload.message, "success");
  } catch (error) {
    setNotice(error.message, "error");
  } finally {
    done();
  }
});

elements.askForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const button = elements.askForm.querySelector("button");
  const done = setBusy(button, "Routing...");
  try {
    const payload = await api("/api/ask", {
      method: "POST",
      body: JSON.stringify({
        query: elements.queryInput.value,
        episodes: Number(elements.episodesInput.value) || 40,
        warm_memory: elements.warmMemoryInput.checked,
      }),
    });
    renderStatus(payload.status);
    renderAskResult(payload.result);
    renderStepGraph(payload.result, payload.status.last_feedback);
    guideToFeedback();
    setNotice(payload.message, "success");
  } catch (error) {
    setNotice(error.message, "error");
  } finally {
    done();
  }
});

elements.feedbackForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const done = setBusy(elements.feedbackSubmit, "Saving feedback...");
  try {
    const payload = await api("/api/feedback", {
      method: "POST",
      body: JSON.stringify({
        rating: Number(elements.feedbackRating.value),
        notes: elements.feedbackNotes.value,
      }),
    });
    renderStatus(payload.status);
    setNotice(payload.message, "success");
  } catch (error) {
    setNotice(error.message, "error");
  } finally {
    done();
  }
});

elements.resetButton.addEventListener("click", async () => {
  const done = setBusy(elements.resetButton, "Resetting...");
  try {
    const payload = await api("/api/reset", {
      method: "POST",
      body: JSON.stringify({}),
    });
    renderStatus(payload.status);
    renderEvaluation([]);
    renderPolicy({ q_table: [], memory_bank: [] });
    elements.evaluationSummary.textContent = "No evaluation yet";
    elements.evaluationDetail.textContent = "Run the benchmark suite to inspect accuracy and rewards.";
    elements.answerSummary.textContent = "No query yet";
    elements.answerDetail.textContent = "Ask a custom question to see agent routing and memory usage.";
    elements.resultRoute.textContent = "Route unavailable";
    elements.answerOutput.textContent = "No custom query has been answered yet.";
    if (elements.feedbackAppliedContainer) elements.feedbackAppliedContainer.style.display = "none";
    state.latestInference = null;
    elements.feedbackNotes.value = "";
    elements.feedbackRating.value = "5";
    renderStepGraph(null, null);
    setNotice(payload.message, "neutral");
  } catch (error) {
    setNotice(error.message, "error");
  } finally {
    done();
  }
});

document.querySelectorAll(".sample-chip").forEach((button) => {
  button.addEventListener("click", () => {
    elements.queryInput.value = button.dataset.query || "";
    elements.queryInput.focus();
  });
});

elements.warmMemoryInput.addEventListener("change", updateQueryComposerState);
updateQueryComposerState();

refreshStatus().catch((error) => setNotice(error.message, "error"));
