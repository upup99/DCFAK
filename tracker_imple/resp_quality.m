function [s, sum_max_val] = resp_quality(R, sum_max_val)

    response = fftshift(R);
    [row, col] = ind2sub(size(response),find(response == max(response(:)), 1));
    max_val = response(row, col);
    sum_max_val(end + 1) = max_val;
    mean_max_val = mean(sum_max_val);
%     disp(sum_max_val)
%     disp((max_val / mean_max_val))
    x1 = min(size(R,2), max(1, col - round(0.05 * size(R,2))));
    x2 = min(size(R,2), max(1, col + round(0.05 * size(R,2))));
    y1 = min(size(R,2), max(1, row - round(0.05 * size(R,1))));
    y2 = min(size(R,2), max(1, row + round(0.05 * size(R,1))));
    M = ones(size(R));
    M(y1:y2, x1:x2) = 0;
    sidelobe = response(M==1);
    mu_s = mean(sidelobe(:));
    sigma_s = std(sidelobe(:));
    s = (max_val - mu_s) / (sigma_s + 0.0001);
    % multiply PSR with maximum value 
    s = s * (max_val / mean_max_val);
    
end

