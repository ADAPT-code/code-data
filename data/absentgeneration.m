
%
clear all
clc
for data_num = 1:21
    absentratio = 0.9;
    get_dnames;
    dnames = [dname];
    tot_data = load([dnames '.mat']);
    data1 = tot_data.X(tot_data.y==0,:);
    [nn,mm] = size(tot_data.X);
    [n,m] = size(data1);
    for i = 1:n
        elementnum = numel(data1(i,:));
        data1(i,randperm(elementnum,floor(absentratio*elementnum))) = nan;
    end
    data= [data1;tot_data.X(tot_data.y==1,:)];
    y = [zeros(n,1);ones(nn-n,1)];
    save(sprintf('%s%d%s',dnames,absentratio*100,'.mat'),'data','y')
end

% 
% clear all
% clc
% for data_num = 1:21
%     absentratio = 0.9;
%     get_dnames;
%     dnames = [dname];
%     tot_data = load([dnames '.mat']);
%     [n,m] = size(tot_data.X);
%     data = tot_data.X;
%     y = tot_data.y;
%     for i = 1:n
%         elementnum = numel(data(i,:));
%         data(i,randperm(elementnum,floor(absentratio*elementnum))) = nan;
%     end
%     save(sprintf('%s%d%s',dnames,absentratio*10,'.mat'),'data','y')
% end
