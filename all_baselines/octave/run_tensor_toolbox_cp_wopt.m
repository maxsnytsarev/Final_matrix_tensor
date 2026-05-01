function run_tensor_toolbox_cp_wopt(config_path)
cfg = jsondecode(fileread(config_path));

addpath(cfg.tensor_toolbox_root);
addpath(fullfile(cfg.tensor_toolbox_root, 'libraries', 'lbfgsb', 'Matlab'));

if isfield(cfg, 'random_state') && ~isempty(cfg.random_state)
    try
        rng(double(cfg.random_state));
    catch
        rand('seed', double(cfg.random_state));
        randn('seed', double(cfg.random_state));
    end
end

data = load(cfg.input_mat);
shape = double(cfg.shape(:))';
X = reshape(double(data.observed_tensor), shape);
W = reshape(double(data.mask) ~= 0, shape);

% CP-WOPT expects X to have zeros in missing entries and W/P as the mask.
X(~W) = 0;

opts = struct();
opts.maxIts = max(1, double(cfg.max_iter));
opts.maxTotalIts = max(opts.maxIts, 10 * opts.maxIts);
opts.printEvery = 0;
opts.pgtol = max(0, double(cfg.tol));

verbosity = 0;
if isfield(cfg, 'verbose') && logical(cfg.verbose)
    verbosity = 10;
    opts.printEvery = 1;
end

[P, ~, out] = cp_wopt( ...
    tensor(X), ...
    tensor(double(W)), ...
    double(cfg.rank), ...
    'skip_zeroing', true, ...
    'verbosity', verbosity, ...
    'opt_options', opts, ...
    'init', 'randn' ...
);

completed = double(full(P));
completed(W) = X(W);
weights = double(P.lambda(:));
factors = P.u;
effective_rank = double(numel(weights));
final_objective = NaN;
iterations = NaN;
optimizer_message = '';
if isfield(out, 'f')
    final_objective = double(out.f);
end
if isfield(out, 'ExitMsg')
    optimizer_message = char(out.ExitMsg);
end
if isfield(out, 'OptOut')
    if isfield(out.OptOut, 'iterations')
        iterations = double(out.OptOut.iterations);
    end
    if isfield(out.OptOut, 'lbfgs_message1')
        optimizer_message = char(out.OptOut.lbfgs_message1);
    end
end

save( ...
    '-mat7-binary', ...
    cfg.output_mat, ...
    'completed', ...
    'weights', ...
    'factors', ...
    'effective_rank', ...
    'final_objective', ...
    'iterations', ...
    'optimizer_message' ...
);
end
