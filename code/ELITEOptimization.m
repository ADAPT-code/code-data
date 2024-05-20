classdef ELITEOptimization < handle

    methods(Static)
        function ELITE = getModel(ELITE)
            objFun = @(parameter) ELITEOptimization.getObjValue(parameter, ELITE);
            ELITE.optimization.numVariables = length(ELITE.optimization.variableName);
            tmp_ = cell(1, ELITE.optimization.numVariables);
            for i = 1:ELITE.optimization.numVariables
                switch ELITE.optimization.variableType{i}
                    case 'real'
                        tmp_{i} = 'double';
                    case 'integer'
                        tmp_{i} = 'int32';
                end
            end

            ELITE.optimization.bestParam = array2table(zeros(1, size(ELITE.optimization.lowerBound, 2)));
            ELITE.optimization.bestParam.Properties.VariableNames = ELITE.optimization.variableName;
            %
            switch ELITE.optimization.method
                case 'bayes'
                    ELITEOptimization.bayesopt(ELITE, objFun);
                case 'ga'
                    ELITEOptimization.gaopt(ELITE, objFun);
                case 'pso'
                    ELITEOptimization.psoopt(ELITE, objFun);
            end
            %
            ELITEOptimization.setParameter(ELITE);
            ELITE.getModel;
        end

        function bayesopt(ELITE, objFun)
            %{
                Optimize the parameters using Bayesian optimization.
                For detailed introduction of the algorithm and parameter
                setting, please enter 'help bayesopt' in the command line.
            %}
            parameter = [];
            % set variable range and type
            var_ = length(ELITE.optimization.lowerBound);
            for i = 1:var_
                tmp_ = optimizableVariable(ELITE.optimization.variableName{i},...
                    [ELITE.optimization.lowerBound(i) ELITE.optimization.upperBound(i)],...
                    'Type', ELITE.optimization.variableType{i});
                parameter = [parameter, tmp_];
            end
            results = bayesopt(objFun, parameter, 'Verbose', 1,...
                'MaxObjectiveEvaluations', ELITE.optimization.maxIteration,...
                'NumSeedPoints', ELITE.optimization.points);
            % optimization results
            [ELITE.optimization.bestParam, ~, ~] = bestPoint(results, 'Criterion', 'min-observed');
        end

        function gaopt(ELITE, objFun)
            %{
                Optimize the parameters using Genetic Algorithm (GA)
                For detailed introduction of the algorithm and parameter
                setting, please enter 'help ga' in the command line.
            %}
            seedSize = 10*ELITE.optimization.numVariables;
            try
                options = optimoptions('ga', 'PopulationSize', seedSize,...
                    'MaxGenerations', ELITE.optimization.maxIteration,...
                    'Display', 'diagnose', 'PlotFcn', 'gaplotbestf');
                bestParam_ = ga(objFun, ELITE.optimization.numVariables, [], [], [], [],...
                    ELITE.optimization.lowerBound, ELITE.optimization.upperBound, [], [], options);
            catch % older vision
                options = optimoptions('ga', 'PopulationSize', seedSize,...
                    'MaxGenerations', ELITE.optimization.maxIteration,...
                    'Display', 'diagnose', 'PlotFcn', @gaplotbestf);
                bestParam_ = ga(objFun, ELITE.optimization.numVariables, [], [], [], [],...
                    ELITE.optimization.lowerBound, ELITE.optimization.upperBound, [], [], options);
            end
            % optimization results
            ELITE.optimization.bestParam.Variables = bestParam_;
        end

        function psoopt(ELITE, objFun)
            %{
                Optimize the parameters using Particle Swarm Optimization (PSO)
                For detailed introduction of the algorithm and parameter
                setting, please enter 'help particleswarm' in the command line.
            %}
            seedSize = 10*ELITE.optimization.numVariables;
            options = optimoptions('particleswarm', 'SwarmSize', seedSize,...
                'MaxIterations', ELITE.optimization.maxIteration,...
                'Display', 'iter', 'PlotFcn', 'pswplotbestf');
            bestParam_ = particleswarm(objFun, ELITE.optimization.numVariables,...
                ELITE.optimization.lowerBound, ELITE.optimization.upperBound, options);
            % optimization results
            ELITE.optimization.bestParam.Variables = bestParam_;
        end

        function objValue = getObjValue(parameter, ELITE)
            %{
                Compute the value of objective function
            %}
            ELITE_ = copy(ELITE);
            ELITE_.display = 'off';
            switch class(parameter)
                case 'table' % bayes
                    ELITE_.optimization.bestParam = parameter;
                case 'double' % ga, pso
                    ELITE_.optimization.bestParam.Variables = parameter;
            end
            % parameter setting
            ELITEOptimization.setParameter(ELITE_);
            % cross validation
            if strcmp(ELITE_.crossValidation.switch, 'on')
                objValue = 1-ELITEOptimization.crossvalFunc(ELITE_);
            else
                % train with all samples
                ELITE_.getModel;
                ELITE_.evaluationMode = 'train';
                results_ = test(ELITE_, ELITE_.data, ELITE_.label);
                ELITE_.performance = ELITE_.evaluateModel(results_);
                objValue = 1-ELITE_.performance.accuracy;
            end
        end



        function setParameter(ELITE)
            %{
                Supported parameter: cost, degree, offset, gamma
            %}
            name_ = ELITE.optimization.bestParam.Properties.VariableNames;
            for i = 1:length(name_)
                switch name_{i}
                    case 'cost' % ELITE parameter
                        ELITE.cost(1) = ELITE.optimization.bestParam.cost;
                    case {'degree', 'offset', 'gamma'} % kernel function parameter
                        ELITE.kernelFunc.(name_{i}) = ELITE.optimization.bestParam.(name_{i});
                end
            end
        end
    end
end