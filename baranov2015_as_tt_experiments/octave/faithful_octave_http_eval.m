function vals = faithful_octave_http_eval(ind, oracle_bridge_dir)
if isempty(ind)
    vals = [];
    return
end

request_dir = fullfile(oracle_bridge_dir, 'requests');
response_dir = fullfile(oracle_bridge_dir, 'responses');
request_path = [tempname(request_dir), '.json'];
[~, request_stem, ~] = fileparts(request_path);
response_path = fullfile(response_dir, [request_stem, '.json']);
cleanup_obj = onCleanup(@() faithful_cleanup_files(request_path, response_path));

payload = struct('indices', double(ind));
fid = fopen(request_path, 'w');
if fid < 0
    error('Faithful:RequestWrite', 'Unable to open request file for the Octave TT-Toolbox bridge.');
end
fprintf(fid, '%s', jsonencode(payload));
fclose(fid);

tries = 0;
while ~exist(response_path, 'file')
    pause(0.01);
    tries = tries + 1;
    if tries > 600000
        error('Faithful:BridgeTimeout', 'Timed out while waiting for the Python TT-Toolbox bridge response.');
    end
end

txt = fileread(response_path);
resp = jsondecode(txt);
if isfield(resp, 'budget_exceeded') && resp.budget_exceeded
    message = 'Budget exceeded in Python oracle bridge.';
    if isfield(resp, 'message')
        message = resp.message;
    end
    error('Faithful:BudgetExceeded', '%s', message);
end
if isfield(resp, 'error')
    error('Faithful:PythonOracleError', '%s', resp.error);
end
if ~isfield(resp, 'values')
    error('Faithful:BadOracleResponse', 'Missing `values` field in Python oracle bridge response.');
end
vals = double(resp.values(:));
end


function faithful_cleanup_files(varargin)
for k = 1:nargin
    path = varargin{k};
    if exist(path, 'file')
        unlink(path);
    end
end
end
