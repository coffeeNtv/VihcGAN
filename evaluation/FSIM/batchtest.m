%clc
%clear
function [fsim,fsimc] = batchtest(ref_path,dis_path)

    ref_list = dir(strcat(ref_path,'*.png'));
    dis_list = dir(strcat(dis_path,'*.png'));
    img_num = length(dis_list);
    sum_fsim = 0;
    sum_fsimc = 0;
    tmp_fsim= 0;
    tmp_fsimc =0;
    fsim_list = [];
    fsimc_list = [];
    if img_num > 0
        for j = 1:img_num
            im1 = ref_list(j).name;
			im2 = dis_list(j).name;
            img1 =  imread(strcat(ref_path,im1));
            img2 =  imread(strcat(dis_path,im2));
    		[tmp_fsim,tmp_fsimc] = FeatureSIM(img1,img2);
            tmp_fsim(find(isnan(tmp_fsim)==1)) = 0;
            tmp_fsimc(find(isnan(tmp_fsimc)==1)) = 0;
            sum_fsim = sum_fsim + tmp_fsim;
            sum_fsimc = sum_fsimc + tmp_fsimc;
        end
    end

    fsim = sum_fsim/img_num
    fsimc = sum_fsimc/img_num
end
