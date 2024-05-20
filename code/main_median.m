clc
clear all
close all
addpath('H:\3. 特征缺失的异常检测\pami_code\data')

res_all = [];
for data_num = 1:21
    get_dnames1;
    dnames = [dname];

    disp('********* load data **********')
    tot_data = load([dnames '.mat']);

    data_only = tot_data.data;
    data_only(isnan(data_only)) = median(data_only(~isnan(data_only)));

    data_labels = tot_data.y;
    data_label = data_labels;

    %% cross fold 5
    count = 0;
    res.dname = [dname];
    outlier_data = data_only(data_label==1,:);
    outlier_data = cat(2,outlier_data,ones(size(outlier_data,1),1).*-1);
    normal_data = data_only(data_label==0,:);
    normal_data = cat(2, normal_data, ones(size(normal_data,1),1));

    [normal_num, normal_dim] = size(normal_data);
    normal_indices=crossvalind('Kfold',normal_data(1:normal_num,normal_dim),5);
    % [outlier_num,outlier_dim] = size(outlier_data);
    % outlier_indices=crossvalind('Kfold',outlier_data(1:outlier_num,outlier_dim),5);

    %kk = 1;
    for f = 1:5
        test_normalind = (normal_indices==f);
        train_normalind =~ test_normalind;
        % test_outlierind = (outlier_indices==f);
        % train_outlierind =~ test_outlierind;

        test_normal = normal_data(test_normalind,:);
        train_normal = normal_data(train_normalind,:);
        % test_outlier = outlier_data(test_outlierind,:);
        % train_outlier = outlier_data(train_outlierind,:);

        disp('********* train test val **********')
        train_data = train_normal(:,1:end-1);
        train_lbls = train_normal(:,end);
        test_data = cat(1,test_normal(:,1:end-1),outlier_data(:,1:end-1));
        test_lbls = cat(1,test_normal(:,end), outlier_data(:,end));

        %sigm_array = power(2,-3:3);
        %nu_array = [0.01:0.05:0.99];
        sigm_array = 2^-3;
        nu_array = 0.01;

        temp_ind = 1;

        for sigm = 1:length(sigm_array)
            for nu = 1:length(nu_array)

                count = count + 1;

                [data_num f sigm_array(sigm) nu_array(nu)]

                disp('********* begin training **********')

                kernel = Kernel('type', 'gaussian', 'gamma', sigm_array(sigm));
                cost = 1/(nu*size(train_data,1));
                ELITEparameter = struct('cost', cost, 'kernelFunc',kernel);

                disp('********* creat an ELITE object **********')
                ELITE = BaseELITE(ELITEparameter);

                ELITE.train(train_data,train_lbls);
                results = ELITE.test(test_data, test_lbls);

                test_acc(temp_ind) = results.performance.accuracy;
                test_auc(temp_ind) = results.performance.AUC;
                test_precision(temp_ind) = results.performance.precision;
                test_recall(temp_ind) = results.performance.recall;
                test_f1score(temp_ind) = results.performance.f1score;
                test_gmean(temp_ind) = results.performance.gmean;
                test_specificity(temp_ind) = results.performance.specificity;
                test_auprc(temp_ind) = results.performance.AUPRC;
                test_time(temp_ind) = results.runningTime;

                temp_ind = temp_ind + 1;
            end
        end
        [max_test,test_ind] = max(test_acc);
        acc(f) = test_acc(test_ind);
        auc(f) = test_auc(test_ind);
        f1score(f) = test_f1score(test_ind);
        gmean(f) = test_gmean(test_ind);
        recall(f) = test_recall(test_ind);
        precision(f) = test_precision(test_ind);
        specificity(f) = test_specificity(test_ind);
        auprc(f) = test_auprc(test_ind);
        time(f) = test_time(test_ind);
        clear test_normalind train_normalind test_normal...
            train_normal test_outlier train_outlier train_data train_lbls test_data test_lbls
    end
    res.testaccfold = acc;
    res.testaucfold = auc;
    res.testf1scorefold = f1score;
    res.testgmeanfold = gmean;
    res.testrecallfold = recall;
    res.testprecisionfold = precision;
    res.testspecificityfold = specificity;
    res.testauprcfold = auprc;
    res.testtimefold = time;

    testacc = mean(acc);
    testaccauc = mean(auc);
    testf1score = mean(f1score);
    testgmean = mean(gmean);
    testrecall = mean(recall);
    testprecision = mean(precision);
    testspecificity = mean(specificity);
    testauprc = mean(auprc);
    testtime = mean(time);
    test_result = [testacc testaccauc testf1score testgmean testrecall testprecision testspecificity testauprc testtime];
    res.testmean = test_result;

    testaccstd = std(acc);
    testaccaucstd = std(auc);
    testf1scorestd = std(f1score);
    testgmeanstd = std(gmean);
    testrecallstd = std(recall);
    testprecisionstd = std(precision);
    testspecificitystd = std(specificity);
    testauprcstd = std(auprc);
    testtimestd = std(time);
    test_resultstd = [testaccstd testaccaucstd testf1scorestd testgmeanstd testrecallstd testprecisionstd testspecificitystd  testauprcstd testtimestd];
    res.teststd = test_resultstd;
    clear   test_result test_resultstd;
    res.performance = {'accuracy', 'auc', 'f1', 'gmean', 'recall', 'precision', 'specificity', 'auprc', 'time'};
    res_all{data_num} = res;
    save(sprintf('%s%d%s','median',data_num,'.mat'),'res_all')
end
save(sprintf('%s%d%s','median_result',9,'.mat'),'res_all');