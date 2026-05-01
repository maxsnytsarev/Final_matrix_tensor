function c = superclasses(~)
%SUPERCLASSES Minimal Octave compatibility shim for Tensor Toolbox objects.
% Octave does not implement MATLAB's superclasses function. Tensor Toolbox
% only needs an empty result for its value classes in our baseline bridge.
c = {};
end
