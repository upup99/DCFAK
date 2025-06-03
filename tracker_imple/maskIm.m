function newIm = maskIm(im, pos, target_sz)
        rect_position = [pos([2,1]) - (target_sz([2,1]) - 1)/2, target_sz([2,1])];
        % 假设 'im' 是输入图像
        
        if size(im,3) == 3
            if all(all(im(:,:,1) == im(:,:,2)))
                im_to_color = false;
            else
                im_to_color = true;
            end
        else
            im_to_color = false;
        end

        % 判断图像的维度
        if im_to_color == false
            % 如果是灰度图像
            im_to_show = im;  % 保持原来的灰度图像
            % 创建一个与图像大小相同的全1矩阵
            mask = ones(size(im));

            % 获取矩形框的参数
            x = rect_position(1);
            y = rect_position(2);
            width = rect_position(3);
            height = rect_position(4);

            % 将矩形区域置为0
            mask(y:(y + height - 1), x:(x + width - 1)) = 0;

            % 应用掩码到图像
            newIm = im_to_show; % 保留原图
            newIm(mask == 0) = 0; % 只把覆盖区域置为0

        else
            % 如果是RGB图像
            im_to_show = im;  % 保持原来的RGB图像
            % 创建一个与图像大小相同的全1矩阵
            mask = ones(size(im_to_show, 1), size(im_to_show, 2));

            % 获取矩形框的参数
            x = rect_position(1);
            y = rect_position(2);
            width = rect_position(3);
            height = rect_position(4);

            % 将矩形区域置为0
            mask(y:(y + height - 1), x:(x + width - 1)) = 0;

            % 应用掩码到图像
            newIm = im_to_show; % 保留原图
            newIm(repmat(mask == 0, [1, 1, size(im_to_show, 3)])) = 0; % 只把覆盖区域置为0
        end

        % 显示结果
        figure;
        imagesc(newIm);
        axis image; % 保持图像比例