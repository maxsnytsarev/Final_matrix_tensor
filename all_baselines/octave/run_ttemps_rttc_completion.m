function run_ttemps_rttc_completion(config_path)
cfg = jsondecode(fileread(config_path));

addpath(cfg.ttemps_root);
addpath(fullfile(cfg.ttemps_root, 'algorithms'));
addpath(fullfile(cfg.ttemps_root, 'algorithms', 'completion'));
addpath(fullfile(cfg.ttemps_root, 'operators'));

if isfield(cfg, 'random_state') && ~isempty(cfg.random_state)
    try
        rng(double(cfg.random_state));
    catch
        rand('seed', double(cfg.random_state));
        randn('seed', double(cfg.random_state));
    end
end

data = load(cfg.input_mat);
n = double(cfg.shape(:))';
d = numel(n);
observed_tensor = reshape(double(data.observed_tensor), n);
mask = reshape(logical(data.mask), n);

idx = find(mask(:));
if isempty(idx)
    error('TTeMPS:NoObservedEntries', 'RTTC requires at least one observed entry.');
end

subs = cell(1, d);
[subs{:}] = ind2sub(n, idx);
Omega = zeros(numel(idx), d);
for k = 1:d
    Omega(:, k) = double(subs{k}(:));
end
A_Omega = observed_tensor(idx);

rr = double(cfg.tt_rank(:))';
X0 = TTeMPS_rand(rr, n);
nrm = norm(X0);
if nrm > 0
    X0 = (1 / nrm) * X0;
end
X0 = orthogonalize(X0, X0.order);

opts = struct();
opts.maxiter = max(1, double(cfg.max_iter));
opts.tol = max(0, double(cfg.tol));
opts.reltol = max(0, double(cfg.tol));
opts.gradtol = max(eps, double(cfg.tol));
opts.cg = true;
opts.verbose = false;
if isfield(cfg, 'verbose')
    opts.verbose = logical(cfg.verbose);
end

% completion_orth is the RTTC implementation used by the TTeMPS examples.
[X, cost, test, stats] = completion_orth(A_Omega, Omega, A_Omega, Omega, X0, opts);
completed = full(X);
completed(mask) = observed_tensor(mask);
cores = X.U;
effective_tt_ranks = double(X.rank(:))';
cost_history = double(cost(:));
test_history = double(test(:));
time_history = [];
converged = false;
if isstruct(stats)
    if isfield(stats, 'time')
        time_history = double(stats.time(:));
    end
    if isfield(stats, 'conv')
        converged = logical(stats.conv);
    end
end

save( ...
    '-mat7-binary', ...
    cfg.output_mat, ...
    'completed', ...
    'cores', ...
    'effective_tt_ranks', ...
    'cost_history', ...
    'test_history', ...
    'time_history', ...
    'converged' ...
);
end
