function yf = peakoptimize(response, y_0, params, disp_row, disp_col, w, reg_sz)
peak=max(response(:));
response=response/max(response(:));
response=circshift(response,floor([size(response,1),size(response,2)]/2));
response=circshift(response,round([-disp_row,-disp_col]));
bg_response=bsxfun(@times,response,w/1e5);

y=circshift(y_0,floor(([size(y_0,1),size(y_0,2)]/2)));
%找局部极大值
%求得响应图的极大值，矩阵对应位置赋值为1，其余部分为0
BW = imregionalmax(response);
Bys=floor(size(BW,1)/2-reg_sz(1)/2):floor(size(BW,1)/2+reg_sz(1)/2);
Bxs=floor(size(BW,2)/2-reg_sz(2)/2):floor(size(BW,2)/2+reg_sz(2)/2);
BW(Bys,Bxs)=0;
% subplot(1,2,1), imshow(response(1)), title('Original Image');
% subplot(1,2,2), imshow(BW(1)), title('Regional Maxima');
%求得二值图像中的连通情况
CC = bwconncomp(BW);
local_max = [max(response(:)) 0];
if length(CC.PixelIdxList) > 1
    local_max = zeros(length(CC.PixelIdxList),1);
    for i = 1:length(CC.PixelIdxList)
        local_max(i) = response(CC.PixelIdxList{i}(1));
    end
    local_max = sort(local_max, 'descend');
end
%筛选排在前30的局部极大值
if length(local_max)<params.local_nums
    num_max=length(local_max);
else
    num_max=params.local_nums;
end
%找到对应坐标，建立新的回归标签
sum_local=0;
% disp(size(response));
% disp("----------");
for i=1:num_max
    %匹配响应图和回归目标峰值，得到对应回归标签的坐标
    [row,col]=find(bg_response==local_max(i));
%     disp(row);
%     disp(col);
%     disp("----------");
    y(row,col) = -params.beta*local_max(i);
%     y(row,col) = 0;
    sum_local=sum_local+local_max(i);
end
avg_local=sum_local/num_max*peak;
% disp(size(y));
% disp("************");
% avg_list(frame)=avg_local;
y=circshift(y,-floor(([size(y,1),size(y,2)]/2)));
% disp(y);
% disp("**************");
% disp(y_0);
y=y*y_0;
yf=fft2(y);