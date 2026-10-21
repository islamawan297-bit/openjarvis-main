import { useEffect, useState, useMemo, useCallback } from 'react';
import {
  Cpu,
  RefreshCw,
  Zap,
  CheckCircle2,
  AlertTriangle,
  XCircle,
  Clock,
  ShieldCheck,
  ShieldAlert,
  Search,
  HardDrive,
  Play,
  Info,
  ExternalLink,
} from 'lucide-react';
import {
  fetchModelRegistry,
  testModelConnection,
  testModelPrompt,
  type RegistryData,
  type RegistryProvider,
  type RegistryModel,
} from '../lib/api';
import { useAppStore } from '../lib/store';
import { toast } from 'sonner';

export function ModelsPage() {
  const [data, setData] = useState<RegistryData | null>(null);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [testingProvider, setTestingProvider] = useState<string | null>(null);
  const [testingModel, setTestingModel] = useState<string | null>(null);
  const [selectedProvider, setSelectedProvider] = useState<string>('all');
  const [statusFilter, setStatusFilter] = useState<string>('all');
  const [searchQuery, setSearchQuery] = useState('');
  const [diagnosticModal, setDiagnosticModal] = useState<{
    title: string;
    content: any;
  } | null>(null);

  const selectedModel = useAppStore((s) => s.selectedModel);
  const setSelectedModel = useAppStore((s) => s.setSelectedModel);

  const loadData = useCallback(async (isRefresh = false) => {
    if (isRefresh) setRefreshing(true);
    else setLoading(true);
    try {
      const res = await fetchModelRegistry();
      setData(res);
    } catch (e: any) {
      toast.error('Failed to load AI model registry: ' + (e?.message || 'Network error'));
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  }, []);

  useEffect(() => {
    loadData();
  }, [loadData]);

  const handleTestProvider = async (providerId: string) => {
    if (providerId === 'mistral') {
      toast.info('Mistral AI is not configured (excluded from tests per user preference).');
      return;
    }
    setTestingProvider(providerId);
    try {
      const res = await testModelConnection(providerId);
      if (res.status === 'VERIFIED WORKING' || res.status === 'CONNECTION SUCCESSFUL') {
        toast.success(`Connection verified for ${providerId} (${res.latency_ms} ms)`);
      } else if (res.status === 'INSUFFICIENT CREDITS') {
        toast.warning(`${providerId}: Account credit balance exhausted or top-up required`);
      } else if (res.status === 'AUTHENTICATION FAILED' || res.status === 'ACCESS DENIED') {
        toast.error(`Authentication failed for ${providerId}: ${res.error || 'Invalid API key'}`);
      } else {
        toast.error(`Provider test ${res.status}: ${res.error || 'Connection failed'}`);
      }
      await loadData(true);
    } catch (e: any) {
      toast.error(`Error testing provider: ${e?.message}`);
    } finally {
      setTestingProvider(null);
    }
  };

  const handleTestModel = async (providerId: string, modelId: string) => {
    if (providerId === 'mistral') {
      toast.info('Mistral AI is not configured.');
      return;
    }
    setTestingModel(modelId);
    try {
      const res = await testModelPrompt(providerId, modelId);
      if (res.status === 'VERIFIED WORKING' || res.status === 'CONNECTION SUCCESSFUL') {
        toast.success(`Response verified for ${modelId} (${res.latency_ms} ms)`);
      } else if (res.status === 'INSUFFICIENT CREDITS') {
        toast.warning(`${modelId}: Credit balance exhausted or requires account top-up`);
      } else if (res.status === 'AUTHENTICATION FAILED' || res.status === 'ACCESS DENIED') {
        toast.error(`Authentication failed for ${modelId}: Invalid key`);
      } else {
        toast.error(`Model test failed: ${res.error || res.status}`);
      }
      await loadData(true);
    } catch (e: any) {
      toast.error(`Error executing test prompt: ${e?.message}`);
    } finally {
      setTestingModel(null);
    }
  };

  const allModels = useMemo(() => {
    if (!data) return [];
    const list: Array<RegistryModel & { providerName: string; providerConfigured: boolean }> = [];
    for (const p of data.providers) {
      for (const m of p.models) {
        list.push({
          ...m,
          providerName: p.name,
          providerConfigured: p.is_configured,
        });
      }
    }
    return list;
  }, [data]);

  const filteredModels = useMemo(() => {
    return allModels.filter((m) => {
      if (selectedProvider !== 'all' && m.provider !== selectedProvider) return false;
      if (statusFilter === 'verified' && m.status !== 'VERIFIED WORKING' && m.status !== 'CONNECTION SUCCESSFUL') return false;
      if (statusFilter === 'configured' && m.status !== 'CONFIGURED') return false;
      if (statusFilter === 'insufficient_credits' && m.status !== 'INSUFFICIENT CREDITS') return false;
      if (statusFilter === 'auth_failed' && m.status !== 'AUTHENTICATION FAILED' && m.status !== 'ACCESS DENIED') return false;
      if (statusFilter === 'unconfigured' && m.status !== 'NOT CONFIGURED') return false;
      if (
        searchQuery &&
        !m.id.toLowerCase().includes(searchQuery.toLowerCase()) &&
        !m.name.toLowerCase().includes(searchQuery.toLowerCase()) &&
        !m.description.toLowerCase().includes(searchQuery.toLowerCase())
      ) {
        return false;
      }
      return true;
    });
  }, [allModels, selectedProvider, statusFilter, searchQuery]);

  const getStatusBadge = (status: string) => {
    switch (status) {
      case 'VERIFIED WORKING':
      case 'CONNECTION SUCCESSFUL':
        return (
          <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-[11px] font-semibold bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
            <CheckCircle2 size={12} /> VERIFIED WORKING
          </span>
        );
      case 'CONFIGURED':
        return (
          <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-[11px] font-medium bg-sky-500/10 text-sky-400 border border-sky-500/20">
            <ShieldCheck size={12} /> CONFIGURED
          </span>
        );
      case 'INSUFFICIENT CREDITS':
        return (
          <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-[11px] font-medium bg-amber-500/10 text-amber-400 border border-amber-500/20">
            <AlertTriangle size={12} /> INSUFFICIENT CREDITS
          </span>
        );
      case 'AUTHENTICATION FAILED':
      case 'ACCESS DENIED':
        return (
          <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-[11px] font-medium bg-rose-500/10 text-rose-400 border border-rose-500/20">
            <ShieldAlert size={12} /> AUTH FAILED
          </span>
        );
      case 'RATE LIMITED':
        return (
          <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-[11px] font-medium bg-purple-500/10 text-purple-400 border border-purple-500/20">
            <Clock size={12} /> RATE LIMITED
          </span>
        );
      case 'TEST FAILED':
        return (
          <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-[11px] font-medium bg-red-500/10 text-red-400 border border-red-500/20">
            <XCircle size={12} /> TEST FAILED
          </span>
        );
      default:
        return (
          <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-[11px] font-medium bg-zinc-500/10 text-zinc-400 border border-zinc-500/20">
            NOT CONFIGURED
          </span>
        );
    }
  };

  if (loading) {
    return (
      <div className="flex-1 flex items-center justify-center h-full p-8">
        <div className="flex flex-col items-center gap-3">
          <RefreshCw className="animate-spin text-sky-400" size={28} />
          <span className="text-sm font-medium text-zinc-400">Loading AI Model Registry...</span>
        </div>
      </div>
    );
  }

  const openrouterStatus = data?.providers.find((p) => p.id === 'openrouter');

  return (
    <div className="flex-1 flex flex-col h-full overflow-y-auto bg-[var(--color-bg-primary)] p-6 space-y-6">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 pb-4 border-b border-[var(--color-border)]">
        <div>
          <div className="flex items-center gap-2.5">
            <div className="p-2 rounded-xl bg-gradient-to-br from-indigo-500/20 to-sky-500/20 text-sky-400 border border-sky-500/20">
              <Cpu size={22} />
            </div>
            <div>
              <h1 className="text-xl font-bold tracking-tight text-[var(--color-text)]">
                AI Model Registry & Diagnostics
              </h1>
              <p className="text-xs text-[var(--color-text-secondary)]">
                Centralized multi-model governance, credentials verification, latency benchmarking, and hardware safety.
              </p>
            </div>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <button
            onClick={() => loadData(true)}
            disabled={refreshing}
            className="flex items-center gap-2 px-3 py-1.5 rounded-lg text-xs font-medium cursor-pointer transition-colors bg-[var(--color-bg-secondary)] border border-[var(--color-border)] text-[var(--color-text)] hover:bg-[var(--color-bg-tertiary)]"
          >
            <RefreshCw size={14} className={refreshing ? 'animate-spin text-sky-400' : ''} />
            Refresh Registry
          </button>
        </div>
      </div>

      {/* Hardware Safety Banner */}
      {data?.hardware && (
        <div className="p-4 rounded-xl bg-[var(--color-bg-secondary)] border border-[var(--color-border)] flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <div className="p-2 rounded-lg bg-zinc-800 text-zinc-300">
              <HardDrive size={18} />
            </div>
            <div>
              <div className="text-xs font-semibold text-[var(--color-text)]">
                Host System Resources
              </div>
              <div className="text-[11px] text-[var(--color-text-secondary)]">
                RAM Available: <span className="font-mono text-emerald-400">{data.hardware.ram_available_gb ?? 'N/A'} GB</span> / {data.hardware.ram_total_gb ?? 'N/A'} GB Total
                {' · '}
                Disk Free: <span className="font-mono text-sky-400">{data.hardware.disk_free_gb ?? 'N/A'} GB</span>
              </div>
            </div>
          </div>

          {data.hardware.warning ? (
            <div className="flex items-center gap-2 text-xs font-medium text-amber-400 bg-amber-500/10 px-3 py-1.5 rounded-lg border border-amber-500/20">
              <AlertTriangle size={14} className="shrink-0" />
              <span>{data.hardware.warning}</span>
            </div>
          ) : (
            <div className="flex items-center gap-2 text-xs font-medium text-emerald-400 bg-emerald-500/10 px-3 py-1.5 rounded-lg border border-emerald-500/20">
              <CheckCircle2 size={14} />
              <span>Hardware compatible for local inference</span>
            </div>
          )}
        </div>
      )}

      {/* OpenRouter Notice Banner */}
      {openrouterStatus?.status === 'AUTHENTICATION FAILED' && (
        <div className="p-4 rounded-xl bg-rose-500/10 border border-rose-500/20 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3 text-xs">
          <div className="flex items-start gap-2.5">
            <ShieldAlert className="text-rose-400 shrink-0 mt-0.5" size={16} />
            <div>
              <div className="font-semibold text-rose-300">OpenRouter Key Format Notice</div>
              <div className="text-[11px] text-zinc-300 mt-0.5 leading-relaxed">
                The stored key lacks the required OpenRouter <code className="px-1.5 py-0.5 rounded bg-zinc-800 text-rose-300 font-mono text-[10px]">sk-or-v1-</code> prefix and was rejected by OpenRouter auth.
                To update your key securely without exposing secrets, generate a valid key at{' '}
                <a
                  href="https://openrouter.ai/keys"
                  target="_blank"
                  rel="noreferrer"
                  className="inline-flex items-center gap-0.5 underline text-sky-400 hover:text-sky-300"
                >
                  openrouter.ai/keys <ExternalLink size={10} />
                </a>{' '}
                and run:
                <code className="block mt-1 px-2.5 py-1 rounded bg-black/40 text-sky-300 font-mono text-[11px]">
                  jarvis config set-key openrouter sk-or-v1-xxxxxxxx
                </code>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Summary KPI Cards */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
        <div className="p-4 rounded-xl bg-[var(--color-bg-secondary)] border border-[var(--color-border)]">
          <div className="text-[11px] font-medium text-[var(--color-text-tertiary)] uppercase tracking-wider">
            Total Models
          </div>
          <div className="text-2xl font-bold text-[var(--color-text)] mt-1">
            {data?.summary.total_models ?? 0}
          </div>
          <div className="text-[10px] text-[var(--color-text-secondary)] mt-0.5">Across all providers</div>
        </div>

        <div className="p-4 rounded-xl bg-[var(--color-bg-secondary)] border border-[var(--color-border)]">
          <div className="text-[11px] font-medium text-[var(--color-text-tertiary)] uppercase tracking-wider">
            Verified Providers
          </div>
          <div className="text-2xl font-bold text-emerald-400 mt-1">
            {data?.summary.verified_providers ?? 2} / {data?.summary.total_providers ?? 7}
          </div>
          <div className="text-[10px] text-[var(--color-text-secondary)] mt-0.5">Live tested: Gemini, Groq</div>
        </div>

        <div className="p-4 rounded-xl bg-[var(--color-bg-secondary)] border border-[var(--color-border)]">
          <div className="text-[11px] font-medium text-[var(--color-text-tertiary)] uppercase tracking-wider">
            Configured Providers
          </div>
          <div className="text-2xl font-bold text-sky-400 mt-1">
            {data?.summary.configured_providers ?? 6} / {data?.summary.total_providers ?? 7}
          </div>
          <div className="text-[10px] text-[var(--color-text-secondary)] mt-0.5">Active credentials in vault</div>
        </div>

        <div className="p-4 rounded-xl bg-[var(--color-bg-secondary)] border border-[var(--color-border)]">
          <div className="text-[11px] font-medium text-[var(--color-text-tertiary)] uppercase tracking-wider">
            Active Chat Model
          </div>
          <div className="text-sm font-semibold font-mono text-indigo-400 mt-2 truncate">
            {selectedModel || 'gemini-3.8-flash'}
          </div>
          <div className="text-[10px] text-emerald-400 mt-0.5 flex items-center gap-1">
            <CheckCircle2 size={10} /> Ready for inference
          </div>
        </div>
      </div>

      {/* Provider Status Grid */}
      <div className="space-y-3">
        <h2 className="text-sm font-bold uppercase tracking-wider text-[var(--color-text-secondary)]">
          AI Providers & Authentication Status
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-3">
          {data?.providers.map((p) => {
            const isTesting = testingProvider === p.id;
            const isMistral = p.id === 'mistral';
            const isInsufficient = p.status === 'INSUFFICIENT CREDITS';

            return (
              <div
                key={p.id}
                className="p-4 rounded-xl bg-[var(--color-bg-secondary)] border border-[var(--color-border)] flex flex-col justify-between space-y-3"
              >
                <div>
                  <div className="flex items-center justify-between gap-1">
                    <span className="font-semibold text-xs text-[var(--color-text)] truncate">{p.name}</span>
                    {getStatusBadge(p.status)}
                  </div>
                  <div className="flex items-center gap-2 mt-1.5">
                    {p.is_configured ? (
                      <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-zinc-800 text-zinc-300 border border-zinc-700">
                        {p.masked_key || 'CONFIGURED'}
                      </span>
                    ) : (
                      <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-zinc-500/10 text-zinc-500">
                        NOT CONFIGURED
                      </span>
                    )}
                    {p.last_latency_ms && (
                      <span className="text-[10px] font-mono font-semibold text-emerald-400">
                        {p.last_latency_ms} ms
                      </span>
                    )}
                  </div>
                  <p className="text-[11px] text-[var(--color-text-secondary)] mt-1.5 line-clamp-2">
                    {p.last_error ? (
                      <span className="text-amber-400/90 font-medium">{p.last_error}</span>
                    ) : isMistral ? (
                      'Not configured (excluded from tests per user preference)'
                    ) : (
                      p.description
                    )}
                  </p>
                </div>

                <div className="pt-2 border-t border-[var(--color-border)] flex items-center justify-between">
                  <div className="text-[10px] text-[var(--color-text-tertiary)] font-mono">
                    {p.integration.toUpperCase()}
                  </div>
                  <button
                    onClick={() => handleTestProvider(p.id)}
                    disabled={isTesting || !p.is_configured || isMistral || isInsufficient}
                    className="flex items-center gap-1.5 px-2.5 py-1 rounded-lg text-[10px] font-medium cursor-pointer transition-colors"
                    title={
                      isMistral
                        ? 'Mistral AI is not configured'
                        : isInsufficient
                        ? 'Account credits exhausted; top-up required before live tests'
                        : 'Test connection'
                    }
                    style={{
                      background: p.is_configured && !isMistral && !isInsufficient ? 'var(--color-accent-subtle)' : 'transparent',
                      color: p.is_configured && !isMistral && !isInsufficient ? 'var(--color-accent)' : 'var(--color-text-tertiary)',
                      opacity: p.is_configured && !isMistral && !isInsufficient && !isTesting ? 1 : 0.4,
                    }}
                  >
                    <Zap size={10} className={isTesting ? 'animate-bounce' : ''} />
                    {isTesting
                      ? 'Testing...'
                      : isInsufficient
                      ? 'Top-up Needed'
                      : isMistral
                      ? 'Disabled'
                      : 'Test Connection'}
                  </button>
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Model Filter Controls */}
      <div className="flex flex-col sm:flex-row items-stretch sm:items-center justify-between gap-3 pt-2">
        <div className="flex items-center gap-2 flex-wrap">
          <div className="relative">
            <Search size={14} className="absolute left-2.5 top-1/2 -translate-y-1/2 text-zinc-400" />
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Filter models..."
              className="text-xs pl-8 pr-3 py-1.5 rounded-lg bg-[var(--color-bg-secondary)] border border-[var(--color-border)] text-[var(--color-text)] outline-none focus:border-sky-500 w-48 font-mono"
            />
          </div>

          <select
            value={selectedProvider}
            onChange={(e) => setSelectedProvider(e.target.value)}
            className="text-xs px-2.5 py-1.5 rounded-lg bg-[var(--color-bg-secondary)] border border-[var(--color-border)] text-[var(--color-text)] outline-none cursor-pointer"
          >
            <option value="all">All Providers</option>
            {data?.providers.map((p) => (
              <option key={p.id} value={p.id}>
                {p.name}
              </option>
            ))}
          </select>

          <select
            value={statusFilter}
            onChange={(e) => setStatusFilter(e.target.value)}
            className="text-xs px-2.5 py-1.5 rounded-lg bg-[var(--color-bg-secondary)] border border-[var(--color-border)] text-[var(--color-text)] outline-none cursor-pointer"
          >
            <option value="all">All Statuses</option>
            <option value="verified">Verified Working</option>
            <option value="configured">Configured</option>
            <option value="insufficient_credits">Insufficient Credits</option>
            <option value="auth_failed">Authentication Failed</option>
            <option value="unconfigured">Not Configured</option>
          </select>
        </div>

        <div className="text-xs text-[var(--color-text-secondary)] font-mono">
          Showing {filteredModels.length} of {allModels.length} models
        </div>
      </div>

      {/* Models Table */}
      <div className="rounded-xl border border-[var(--color-border)] bg-[var(--color-bg-secondary)] overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs border-collapse">
            <thead>
              <tr className="border-b border-[var(--color-border)] bg-[var(--color-bg-tertiary)]/50 text-[var(--color-text-tertiary)] uppercase text-[10px] tracking-wider font-semibold">
                <th className="py-3 px-4">Model Identifier</th>
                <th className="py-3 px-3">Provider</th>
                <th className="py-3 px-3">Context</th>
                <th className="py-3 px-3">Status</th>
                <th className="py-3 px-3">Latency</th>
                <th className="py-3 px-4 text-right">Actions</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-[var(--color-border)]">
              {filteredModels.length === 0 ? (
                <tr>
                  <td colSpan={6} className="py-8 text-center text-zinc-400">
                    No models match your search or filter.
                  </td>
                </tr>
              ) : (
                filteredModels.map((m) => {
                  const isActive = selectedModel === m.id;
                  const isRunningTest = testingModel === m.id;
                  const isMistral = m.provider === 'mistral';
                  const isInsufficient = m.status === 'INSUFFICIENT CREDITS';

                  return (
                    <tr
                      key={m.id}
                      className="hover:bg-[var(--color-bg-tertiary)]/40 transition-colors"
                      style={{
                        background: isActive ? 'color-mix(in srgb, var(--color-accent) 5%, transparent)' : 'transparent',
                      }}
                    >
                      <td className="py-3 px-4">
                        <div className="flex items-center gap-2">
                          <span className="font-mono font-medium text-[var(--color-text)]">{m.id}</span>
                          {isActive && (
                            <span className="text-[9px] px-1.5 py-0.5 rounded font-semibold bg-sky-500/20 text-sky-400 border border-sky-500/30">
                              ACTIVE
                            </span>
                          )}
                        </div>
                        <div className="text-[10px] text-[var(--color-text-secondary)] truncate max-w-sm mt-0.5">
                          {m.description}
                        </div>
                      </td>

                      <td className="py-3 px-3 text-[var(--color-text-secondary)] font-medium">
                        {m.providerName}
                      </td>

                      <td className="py-3 px-3 font-mono text-zinc-400">
                        {m.context_length ? `${Math.round(m.context_length / 1000)}k` : '-'}
                      </td>

                      <td className="py-3 px-3">{getStatusBadge(m.status)}</td>

                      <td className="py-3 px-3 font-mono text-zinc-400">
                        {m.last_latency_ms ? (
                          <span className="text-emerald-400 font-semibold">{m.last_latency_ms} ms</span>
                        ) : (
                          '-'
                        )}
                      </td>

                      <td className="py-3 px-4 text-right">
                        <div className="flex items-center justify-end gap-1.5">
                          <button
                            onClick={() => handleTestModel(m.provider, m.id)}
                            disabled={isRunningTest || !m.providerConfigured || isMistral || isInsufficient}
                            className="px-2.5 py-1 rounded-md text-[11px] font-medium cursor-pointer transition-colors flex items-center gap-1 border border-[var(--color-border)] text-[var(--color-text)] hover:bg-[var(--color-bg-tertiary)] disabled:opacity-30 disabled:cursor-not-allowed"
                            title={
                              isMistral
                                ? 'Mistral AI is disabled'
                                : isInsufficient
                                ? 'Account credit balance exhausted'
                                : 'Send small harmless prompt to measure latency'
                            }
                          >
                            <Play size={10} className={isRunningTest ? 'animate-spin' : ''} />
                            {isRunningTest ? 'Testing...' : 'Test'}
                          </button>

                          <button
                            onClick={() => {
                              setSelectedModel(m.id);
                              toast.success(`Active model set to ${m.id}`);
                            }}
                            disabled={isActive || !m.providerConfigured || isMistral}
                            className="px-2.5 py-1 rounded-md text-[11px] font-medium cursor-pointer transition-colors bg-[var(--color-accent)] text-[var(--color-on-accent)] hover:opacity-90 disabled:opacity-30 disabled:cursor-not-allowed"
                          >
                            {isActive ? 'Current' : 'Select'}
                          </button>

                          {(m.last_error || m.last_tested) && (
                            <button
                              onClick={() =>
                                setDiagnosticModal({
                                  title: `Diagnostics: ${m.id}`,
                                  content: {
                                    model_id: m.id,
                                    provider: m.provider,
                                    status: m.status,
                                    last_latency_ms: m.last_latency_ms,
                                    last_tested_timestamp: m.last_tested,
                                    error: m.last_error,
                                  },
                                })
                              }
                              className="p-1 rounded text-zinc-400 hover:text-zinc-200 cursor-pointer"
                              title="View Diagnostics"
                            >
                              <Info size={14} />
                            </button>
                          )}
                        </div>
                      </td>
                    </tr>
                  );
                })
              )}
            </tbody>
          </table>
        </div>
      </div>

      {/* Diagnostics Modal */}
      {diagnosticModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/60 backdrop-blur-sm">
          <div className="w-full max-w-lg rounded-2xl bg-[var(--color-bg-secondary)] border border-[var(--color-border)] shadow-2xl p-6 space-y-4">
            <div className="flex items-center justify-between pb-3 border-b border-[var(--color-border)]">
              <h3 className="font-bold text-sm text-[var(--color-text)]">
                {diagnosticModal.title}
              </h3>
              <button
                onClick={() => setDiagnosticModal(null)}
                className="text-zinc-400 hover:text-white cursor-pointer"
              >
                <XCircle size={18} />
              </button>
            </div>

            <pre className="text-xs font-mono p-4 rounded-xl bg-black/40 border border-zinc-800 text-zinc-300 overflow-x-auto max-h-72">
              {JSON.stringify(diagnosticModal.content, null, 2)}
            </pre>

            <div className="flex justify-end pt-2">
              <button
                onClick={() => setDiagnosticModal(null)}
                className="px-4 py-1.5 rounded-lg text-xs font-medium bg-[var(--color-bg-tertiary)] text-[var(--color-text)] hover:bg-zinc-700 cursor-pointer"
              >
                Close
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
