%% Tianyang Xu, Zhen-Hua Feng, Xiao-Jun Wu, Josef Kittler. Joint Group
%% Feature Selection and Discriminative Filter Learning for Robust Visual
%% Object Tracking. In ICCV 2019.

function results = tracker_main(params)
% Run tracker
% Input  : params[structure](parameters and sequence information)
% Output : results[structure](tracking results)

% Get sequence information
% <
[seq, im] = get_sequence_info(params.seq);
params = rmfield(params, 'seq');

if isempty(im)
    seq.rect_position = [];
    [~, results] = get_sequence_results(seq);
    return;
end

seq = get_sequence_vot(seq);

pos = seq.init_pos(:)';
target_sz = seq.init_sz(:)';
ratio_sz = target_sz(1)/target_sz(2);
params.init_sz = target_sz;
features = params.t_features;
params = init_default_params(params);
init_target_sz = target_sz;

% >

% Set Global parameters and data type
% <
if isfield(params, 't_global')
    global_fparams = params.t_global;
else
    global_fparams = [];
end
global_fparams.use_gpu = params.use_gpu;
global_fparams.gpu_id = params.gpu_id;
global_fparams.augment = 0;

% 新增
global_fparams.cell_size = 4;

if params.use_gpu
    if isempty(params.gpu_id)
        gD = gpuDevice();
    elseif params.gpu_id > 0
        gD = gpuDevice(params.gpu_id);
    end
    params.data_type = zeros(1, 'single', 'gpuArray');
else
    params.data_type = zeros(1, 'single');
end
params.data_type_complex = complex(params.data_type);

global_fparams.data_type = params.data_type;
% >

% Check if color image
% <
if size(im,3) == 3
    if all(all(im(:,:,1) == im(:,:,2)))
        is_color_image = false;
    else
        is_color_image = true;
    end
else
    is_color_image = false;
end

if size(im,3) > 1 && is_color_image == false
    im = im(:,:,1);
end
% >

% Calculate search area and initial scale factor
% <
search_area = prod(params.search_area_scale'*init_target_sz,2);
currentScaleFactor = zeros(numel(search_area),1);
for i = 1 : numel(currentScaleFactor)
    if search_area(i) > params.max_image_sample_size(i)
        currentScaleFactor(i) = sqrt(search_area(i) / params.max_image_sample_size(i));
    elseif search_area(i) < params.min_image_sample_size(i)
        currentScaleFactor(i) = sqrt(search_area(i) / params.min_image_sample_size(i));
    else
        currentScaleFactor(i) = 1.0;
    end
end
% >

% target size at the initial scale
% <
featureRatio = global_fparams.cell_size;
base_target_sz = 1 ./ currentScaleFactor*target_sz;
reg_sz= floor(base_target_sz/featureRatio);
% >


% search window size
% <
img_sample_sz = repmat(sqrt(prod(base_target_sz,2) .* (params.search_area_scale.^2')),1,2); % square area, ignores the target aspect ratio
% >

% initialise feature settings
% <
[features, global_fparams, feature_info] = init_features(features, global_fparams, params, is_color_image, img_sample_sz, 'odd_cells');
img_support_sz = feature_info.img_support_sz;
feature_sz = feature_info.data_sz;
num_feature_blocks = size(feature_sz, 1);
% >


% <
feature_sz_cell = permute(mat2cell(feature_sz, ones(1,num_feature_blocks), 2), [2 3 1]);
filter_sz = feature_sz + mod(feature_sz+1, 2);
filter_sz_cell = permute(mat2cell(filter_sz, ones(1,num_feature_blocks), 2), [2 3 1]);

[output_sz, k1] = max(filter_sz, [], 1);
params.output_sz = output_sz;
k1 = k1(1);
block_inds = 1:num_feature_blocks;
block_inds(k1) = [];
% >

% Pre-computes the grid that is used for socre optimization
% <
ky = circshift(-floor((filter_sz_cell{1}(1) - 1)/2) : ceil((filter_sz_cell{1}(1) - 1)/2), [1, -floor((filter_sz_cell{1}(1) - 1)/2)]);
kx = circshift(-floor((filter_sz_cell{1}(2) - 1)/2) : ceil((filter_sz_cell{1}(2) - 1)/2), [1, -floor((filter_sz_cell{1}(2) - 1)/2)])';
newton_iterations = params.newton_iterations;
% >

% Construct the Gaussian label function, cosine window and initial mask
% <
yf = cell(num_feature_blocks, 1);
for i = 1:num_feature_blocks
    sz = filter_sz_cell{i};
    output_sigma_factor = params.output_sigma_factor(feature_info.feature_is_deep(i)+1);
    output_sigma  = sqrt(prod(floor(base_target_sz(feature_info.feature_is_deep(i)+1,:))))*feature_sz_cell{i}./img_support_sz{i}* output_sigma_factor;
    rg            = circshift(-floor((sz(1)-1)/2):ceil((sz(1)-1)/2), [0 -floor((sz(1)-1)/2)]);
    cg            = circshift(-floor((sz(2)-1)/2):ceil((sz(2)-1)/2), [0 -floor((sz(2)-1)/2)]);
    [rs, cs]      = ndgrid(rg,cg);
    y_0{i}             = exp(-0.5 * (((rs.^2 + cs.^2) / mean(output_sigma)^2)));
    
    yf{i} = fft2(y_0{i});
end
% yf{1} = zeros(size(yf{1}));
% yf{2} = zeros(size(yf{2}));
% yf{3} = zeros(size(yf{3}));

% disp(size(y_0{1}));

% if params.use_gpu
%     params.data_type = zeros(1, 'single', 'gpuArray');
% else
%     params.data_type = zeros(1, 'single');
% end
% params.data_type_complex = complex(params.data_type);
global_fparams.data_type = params.data_type;

%cos_window = cellfun(@(sz) hann(sz(1)+2)*hann(sz(2)+2)', feature_sz_cell, 'uniformoutput', false);
%cos_window = cellfun(@(sz) single(kaiser(sz(1)+2,7)*kaiser(sz(2)+2,7)'), feature_sz_cell, 'uniformoutput', false);
%cos_window = cellfun(@(cos_window) cast(cos_window(2:end-1,2:end-1), 'like', params.data_type), cos_window, 'uniformoutput', false);

mask_window = cell(1,1,num_feature_blocks);
mask_search_window = cellfun(@(sz,feature) ones(round(currentScaleFactor(feature.fparams.feature_is_deep+1)*sz)) * 1e-3, img_support_sz', features, 'uniformoutput', false);
target_mask = 1.2*seq.target_mask;
target_mask_range = zeros(2, 2);
for j = 1:2
    target_mask_range(j,:) = [0, size(target_mask,j) - 1] - floor(size(target_mask,j) / 2);
end
for i = 1:num_feature_blocks
    mask_center = floor((size(mask_search_window{i}) + 1)/ 2) + mod(size(mask_search_window{i}) + 1,2);
    target_h = (mask_center(1)+ target_mask_range(1,1)) : (mask_center(1) + target_mask_range(1,2));
    target_w = (mask_center(2)+ target_mask_range(2,1)) : (mask_center(2) + target_mask_range(2,2));
    mask_search_window{i}(target_h, target_w) = target_mask;
    mask_window{i} = mexResize(mask_search_window{i}, filter_sz_cell{i}, 'auto');
end
params.mask_window = mask_window;
% >

% Use the pyramid filters to estimate the scale
% <
nScales = params.number_of_scales;
scale_step = params.scale_step;
scale_exp = (-floor((nScales-1)/2):ceil((nScales-1)/2));
scaleFactors = scale_step .^ scale_exp;
% >

% ------add----
% <
for i = 1:num_feature_blocks
    obj_window{i} = yf{i} * 0;                   % obj_window is the crop matrix 
    [sx,sy,~] = get_subwindow_no_window(obj_window{i}, floor(filter_sz_cell{i}/2) , filter_sz_cell{i});
    seq.time = 0;
    mask_c{i} = yf{i} * 0;
%     mask_b{i} = yf{i} * 0 + 1;
    if i == 1
%         mask_b{i}(sx,sy) = 0;
        mask_c{i}(sx,sy) = 1;
    end
    
end
% >

% template
% <
template_filter = [];
% >

% Initialize motion estimotion and failure correction modules
dx_pred = 0;
dy_pred = 0;
dx_true = 0;
dy_true = 0;
mean_max_val = [];

resp_budg = params.resp_budg;
resp_norm = params.resp_norm;
tracking_state = params.tracking_state;  
skip_check_beginning = params.skip_check_beginning;   
uncertainty_thre = params.uncertainty_thre;    
resp_budg_sz = params.resp_budg_sz;   
quality = 0;
minQuality = 100;
maxQuality = 0;
omega = 0;
history_scores = [];
tracking_state_learning = 1;
% >

seq.time = 0;
scores_fs_feat = cell(1,1,num_feature_blocks);
response_feat = cell(1,1,num_feature_blocks);
% response_peak = cell(1,1,num_feature_blocks); % 新增
% tmp = cell(1,1,num_feature_blocks);
store_filter = [];
noise = 7;
while true
    % Read image and timing
    % <
    cos_window = cellfun(@(sz) single(kaiser(sz(1)+2,noise)*kaiser(sz(2)+2,noise)'), feature_sz_cell, 'uniformoutput', false);
    cos_window = cellfun(@(cos_window) cos_window(2:end-1,2:end-1), cos_window, 'uniformoutput', false);
    if seq.frame > 0
        [seq, im] = get_sequence_frame(seq);
        if isempty(im)
            break;
        end
        if size(im,3) > 1 && is_color_image == false
            im = im(:,:,1);
        end
    else
%         for i = 1:num_feature_blocks
%             sz = filter_sz_cell{i};
            [range_h,range_w,win] = init_regwindow(output_sz,reg_sz,params);
%             [range_h,range_w,win{i}] = init_regwindow(sz,reg_sz,params);
%         end
%         disp(size(win{2}));
        seq.frame = 1;
        filter_model_f = [];
        Sfilter_pre_f = [];
    end
    
    tic();
    % >
    if seq.frame == 2
        template_filter_first = filter_model_f;
        template_filter = template_filter_first;
    end
    
%     particle_s1= [0, 0];
%     particle_s2= [0, 0];
    %% Target localization step
    if seq.frame > 1
        global_fparams.augment = 0;
        old_pos = inf(size(pos));
        iter = 1;
        
        % template select
        if seq.frame > 2
            if tracking_state == 2
                filter_model_f = template_filter;
            else
                if tracking_state == 1
                    filter_model_f{1} = bsxfun(@times, tracking_state_learning, template_filter{1}) + bsxfun(@times, 1 - tracking_state_learning , filter_model_f{1});
                    filter_model_f{2} = bsxfun(@times, tracking_state_learning, template_filter{2}) + bsxfun(@times, 1 - tracking_state_learning , filter_model_f{2});
                    filter_model_f{3} = bsxfun(@times, tracking_state_learning, template_filter{3}) + bsxfun(@times, 1 - tracking_state_learning , filter_model_f{3});
                    template_filter = filter_model_f;
                end
            end
        end
        
        %translation search
        while iter <= params.refinement_iterations && any(old_pos ~= pos)
            sample_pos = round(pos);
            sample_scale = currentScaleFactor*scaleFactors;
            [xt, img_samples] = extract_features(im, sample_pos, sample_scale, features, global_fparams, feature_info);
            xtw = cellfun(@(feat_map, cos_window) bsxfun(@times, feat_map, cos_window), xt, cos_window, 'uniformoutput', false);
            xtf = cellfun(@fft2, xtw, 'uniformoutput', false);
            
            % visualize xt
            % <
%             disp(size(xt));
%             for i = 1:numel(xt)
%                 thisVal = xt{i};
%                 newVal = zeros(size(thisVal));
%                 for j = 1:size(thisVal,3)
%                     newVal(:,:,j) = abs(gather(thisVal(:,:,j)));
%                 end
%                 xt{i} = newVal;
%             end
%             
%             numPlots = numel(xt);
%             for i = 1:numPlots
% %                 subplot(1,numPlots,i);
%                 figure;
%                 imagesc(xt{i}(:,:,1));
%                 colorbar;
%                 title(['Visualization of Cell ', num2str(i)]);
%             end
            % >
            
            % Calculate and fuse responses
            % <
            response_handcrafted = 0;
%             response_handpeak = 0;
            response_deep = 0;
            for k = [k1 block_inds]
                if feature_info.feature_is_deep(k) == 0
                    scores_fs_feat{k} = gather(sum(bsxfun(@times, conj(filter_model_f{k}), xtf{k}), 3));
                    scores_fs_feat{k} = resizeDFT2(scores_fs_feat{k}, output_sz);
                    response_feat{k} = ifft2(scores_fs_feat{k}, 'symmetric');
                    response_handcrafted = response_handcrafted + response_feat{k};
                else
                    output_sz_deep = round(output_sz/img_support_sz{k1}*img_support_sz{k});
                    output_sz_deep = output_sz_deep + 1 + mod(output_sz_deep,2);
                    scores_fs_feat{k} = gather(sum(bsxfun(@times, conj(filter_model_f{k}), xtf{k}), 3));
                    scores_fs_feat{k} = resizeDFT2(scores_fs_feat{k}, output_sz_deep);
                    response_feat{k} = ifft2(scores_fs_feat{k}, 'symmetric');
                    response_feat{k}(ceil(output_sz(1)/2)+1:output_sz_deep(1)-floor(output_sz(1)/2),:,:,:)=[];
                    response_feat{k}(:,ceil(output_sz(2)/2)+1:output_sz_deep(2)-floor(output_sz(2)/2),:,:)=[];
                    response_deep = response_deep + response_feat{k};
                end
            end
            [disp_row, disp_col, sind, ~, response, w] = resp_newton(squeeze(response_handcrafted)/feature_info.feature_hc_num, squeeze(response_deep)/feature_info.feature_deep_num,...
                newton_iterations, ky, kx, output_sz);
            % Compute the translation vector
            % <
           skew = kurtosis(response,1,'all');
           if seq.frame == 2
               skew_first = skew;
           else
               noise = 0.1 * noise + 0.9 * max(7 * skew / skew_first, 7.1);
           end

            %---
            translation_vec = [disp_row, disp_col] .* (img_support_sz{k1}./output_sz) * currentScaleFactor(1) * scaleFactors(sind);
            if seq.frame < 10
                scale_change_factor = scaleFactors(ceil(params.number_of_scales/2));
            else
                scale_change_factor = scaleFactors(sind);
            end
            % >
            
            % update position
            % <
            %predict(KF);
            old_pos = pos;
            if sum(isnan(translation_vec))
                pos = sample_pos;
            else
                    pos = sample_pos + translation_vec;
            end
            % failure detection and correction
            
            [quality, mean_max_val] = resp_quality(response, mean_max_val);   
            
            if numel(resp_budg) >= skip_check_beginning
                response_budget_mean = mean(resp_budg);
                curr_quality_norm = quality / resp_norm;
                curr_score = (response_budget_mean - curr_quality_norm) / curr_quality_norm;
                history_scores(end + 1) = curr_quality_norm / response_budget_mean;
                his_len = length(history_scores);
                if his_len > 3
                    if history_scores(his_len - 2) - history_scores(his_len - 1) > 0.2 && history_scores(his_len - 1) - history_scores(his_len) > 0.2
                        tracking_state_learning = 0.9;
                    elseif history_scores(his_len - 1) - history_scores(his_len - 2) > 0.2 && history_scores(his_len) - history_scores(his_len - 1) > 0.2
                        tracking_state_learning = 0.1;
                    else
                        tracking_state_learning = 1 - history_scores(end);
                    end
                end
                        
                if curr_score > uncertainty_thre
                    tracking_state = 2;  % Failure tracking
                else
                    tracking_state = 1;  
                end
            else
                tracking_state = 1; 
            end 
            
            if params.clamp_position
                pos = max([1 1], min([size(im,1) size(im,2)], pos));
            end
            % >
            % Update the scale
            % <
            currentScaleFactor = currentScaleFactor * scale_change_factor;
            % >

			% construct masks
            for i = 1:num_feature_blocks
                response_pre_window = circshift(response(:,:,sind), -floor(output_sz(1:2)/2)+1);           
                max_M_curr = max(response_pre_window(:));
                [id_xmax_curr, id_ymax_curr] = find(response_pre_window == max_M_curr);     
                shift_y = id_ymax_curr - (ceil(output_sz(1)/2)+1);
                shift_x = id_xmax_curr - (ceil(output_sz(2)/2)+1);
                mask_c{i} = yf{i} * 0;
                if i == 1
                    if min(sx+shift_x)<=0 || min(sy+shift_y)<=0 || max(sx+shift_x) > max(size(mask_c{i})) || max(sy+shift_y) > max(size(mask_c{i}))
% %                         mask_b{i}(sx,sy) = params.b_amplify_center;
                        mask_c{i}(sx,sy) = params.c_amplify;
                    else
% %                         mask_b{i}(sx+shift_x,sy+shift_y) = params.b_amplify_center;
                        mask_c{i}(sx+shift_x,sy+shift_y) = params.c_amplify;
                    end
%                     mask_c{i}(sx,sy) = params.c_amplify;
                end
            end
            
            % < 可视化response
%             disp(response(:,:,2))
%             disp(size(response));
%             R = sum(response, 3);
%             figure;
%             surf(fftshift(R(:, :)));
%             dataname = ['D:\CFTrackers-SUM\论文图表\singer2_response\' num2str(seq.frame) '.png'];
%             saveas(gcf, dataname);
            % >

            iter = iter + 1;
        end
        response_peak1 = sum(response, 3);
        yf{1} = peakoptimize(response_peak1, y_0{1}, params, disp_row, disp_col, win, reg_sz); 
    end
    
    % Record the historical information of the response map
    if tracking_state == 1 && seq.frame > 1
        if isempty(resp_budg)
            resp_norm = quality;
            resp_budg(end+1) = 1;
        else
            resp_budg(end+1) = quality / resp_norm;
            if numel(resp_budg) > resp_budg_sz
                resp_budg(1) = [];
            end
        end
    end
    
    %% Model update step
    % Extract features and learn filters
    % <
    global_fparams.augment = 1;
    sample_pos = round(pos);
    [xl, ~] = extract_features(im, sample_pos, currentScaleFactor, features, global_fparams, feature_info);
    xlw = cellfun(@(feat_map, cos_window) bsxfun(@times, feat_map, cos_window), xl, cos_window, 'uniformoutput', false);
    xlf = cellfun(@fft2, xlw, 'uniformoutput', false);

    mask_c_cell = reshape(mask_c, 1, 1, num_feature_blocks);
%         xb_cell = cellfun(@(feat_map, mask_b_cell) bsxfun(@times, feat_map, mask_b_cell), xl, mask_b_cell, 'uniformoutput', false);
    xc_cell = cellfun(@(feat_map, mask_c_cell) bsxfun(@times, feat_map, mask_c_cell), xl, mask_c_cell, 'uniformoutput', false);

%         xb_fft = cellfun(@fft2, xb_cell, 'uniformoutput', false);
    xc_fft = cellfun(@fft2, xc_cell, 'uniformoutput', false);

    
    [filter_model_f,spatial_units, channels] = train_filter2(xlf, feature_info, yf, seq, params, filter_model_f, xc_fft); 
%     [filter_model_f,spatial_units, channels] = train_filter(xlf, feature_info, yf, seq, params, filter_model_f); 
    
    % Update the target size
    % <
    target_sz = base_target_sz(1,:) * currentScaleFactor(1);
    % >
    
    %save position and time
    % <
    tracking_result.center_pos = double(pos);
    tracking_result.target_size = double(target_sz);
    seq = report_tracking_result(seq, tracking_result);
    seq.time = seq.time + toc();
    % >
    
    % visualisation
    % <
    if params.vis_res
        rect_position_vis = [pos([2,1]) - (target_sz([2,1]) - 1)/2, target_sz([2,1])];
%         disp(rect_position_vis)
        im_to_show = double(im)/255;
%         disp(size(im_to_show));
        if size(im_to_show,3) == 1
            im_to_show = repmat(im_to_show, [1 1 3]);
        end
        if seq.frame == 1  %first frame, create GUI
            fig_handle = figure('Name', 'Tracking','Position',[100, 300, 600, 480]);
            set(gca, 'position', [0 0 1 1 ]);
            axis off;axis image;
            imagesc(im_to_show);
            hold on;
            rectangle('Position',rect_position_vis, 'EdgeColor','g', 'LineWidth',2);
            text(10, 10, int2str(seq.frame), 'color', [0 1 1], 'FontSize',20);
            hold off;
            axis off;axis image;set(gca, 'Units', 'normalized', 'Position', [0 0 1 1])
        else
            figure(fig_handle);
            imagesc(im_to_show);
            hold on;
            rectangle('Position',rect_position_vis, 'EdgeColor','g', 'LineWidth',2);
            text(10, 10, int2str(seq.frame), 'color', [0 1 1], 'FontSize',20);
            hold off;
        end

        drawnow
    end
    
    if params.vis_res && params.vis_details
        if seq.frame == 1
            res_vis_sz = [300, 480];
            fig_handle_detail = figure('Name', 'Details','Position',[700, 300, res_vis_sz]);
            anno_handle = annotation('textbox',[0.3,0.92,0.7,0.08],'LineStyle','none',...
                'String',['Tracking Frame #' num2str(seq.frame)]);
        else
            figure(fig_handle_detail);
            set(anno_handle, 'String',['Tracking Frame #' num2str(seq.frame)]);
            set(gca, 'position', [0 0 1 1 ]);
            subplot('position',[0.05,0.08,0.9,0.12]);
            bar(channels{end}(:)>0);
            xlabel(['channels [' num2str(params.channel_selection_rate(2)) ']']);
            title('Selected Channels for Deep features');
            subplot('position',[0.05,0.3,0.4,0.25]);
            imagesc(spatial_units{1}>0);
            xlabel(['spatial units [' num2str(params.spatial_selection_rate(1)) ']']);
            title('Hand-crafted features');
            subplot('position',[0.55,0.3,0.4,0.25]);
            imagesc(spatial_units{end}>0);
            xlabel(['spatial units [' num2str(params.spatial_selection_rate(2)) ']']);
            title('Deep features');
            subplot('position',[0.05,0.65,0.4,0.25]);
            patch_to_show = imresize(img_samples{1}(:,:,:,sind),res_vis_sz);
            imagesc(patch_to_show);
            hold on;
            sampled_scores_display = circshift(imresize(response(:,:,sind),...
                res_vis_sz),floor(0.5*res_vis_sz));
            resp_handle = imagesc(sampled_scores_display);
            alpha(resp_handle, 0.6);
            hold off;
            axis off;
            title('Response map');
            subplot('position',[0.55,0.65,0.4,0.25]);
            bar([w,1-w]);axis([0.5 2.5 0 1]);
            set(gca,'XTickLabel',{'HC','Deep'});
            title('Weights HC vs. Deep');
        end
    end
    % >
    
    % store filters for final rank calculation 
    % (can be commented on benchmark experiments)
    % (please comment the following lines if sequence frames > 3000)
    % < %%
    temp = [];
    for tt = 1 : num_feature_blocks
        temp = [temp gather(filter_model_f{tt}(:)')];
    end
    store_filter = [store_filter temp'];
    % > %%
end
% get tracking results
% <
[~, results] = get_sequence_results(seq); 
if params.vis_res&& params.vis_details
    close(fig_handle_detail);close(fig_handle);
elseif params.vis_res
    close(fig_handle);
end
% >

% rank calculation for the entire sequence
% (can be commented on benchmark experiments)
% < %%
if ~isempty(store_filter)
    results.rank_var = rank(store_filter); 
else
    results.rank_var = -1;
end
% > %%