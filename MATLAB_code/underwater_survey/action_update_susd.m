function agent_self = action_update_susd(agent, P, Y, hyperparams, N_steps, max_num_task_selected)

    [N_agents, N_tasks] = size(agent.A);
    % action = task selection
    N_param = N_tasks*2;
    agent_id = agent.id;
    original_agent = agent; 
    utility_org = utility_calculation(original_agent, P, Y);
    utilty_best = utility_org;
    best_agent = original_agent;
    best_theta = original_agent.theta;
    
    null_agent = original_agent;
    null_agent.theta = zeros(1, 2*N_tasks);
    null_agent.Theta(agent_id,:) = null_agent.theta;
    null_assignments = zeros(N_tasks, 1);
    null_agent.A(agent_id,:) = null_assignments;
    null_agent.assignments = null_assignments;
    null_agent.selected_tasks = find(null_assignments);
    utility_null = utility_calculation(null_agent, P, Y);

    if utility_null > utilty_best
        best_agent = null_agent;
        best_theta = null_agent.theta;
        agent = null_agent;
    end

    if ~exist('max_num_task_selected','var')
     % third parameter does not exist, so default it to something
      max_num_task_selected = agent.max_num_tasks;
    end
    

    %% SUSD approach 
    k1 = hyperparams(1); 
    k2 = hyperparams(2); 
    d0 = hyperparams(3); 
    eps = hyperparams(4);
    N_vagents = 2*N_param;  

    % create virtual agents
    % n_ = zeros(N_param,1); n_(1) = 1;
    theta_ = zeros(N_param, N_vagents);
    vagents = agent([]);
    for v_i=1:N_vagents
        theta_(:,v_i) = agent.theta'+ 0.2*rand(N_param,1);
        vagents(v_i) = agent;
    end
    theta_(:,1) = agent.theta';

    % perform optimization loop
    for t = 1:N_steps
        Cx = zeros(1,N_vagents);
        for v_i=1:N_vagents
            theta_vi = theta_(:, v_i)';
            vagents(v_i).theta = theta_vi;
            vagents(v_i).Theta(agent_id,:) = theta_vi;
            % curr_utility = evaluation_from_theta(vagents(v_i), P, Y, 'L_largest', 1);
            assignments = theta_to_assignment(vagents(v_i).theta, max_num_task_selected, 'L_largest');
            vagents(v_i).assignments = assignments;
            vagents(v_i).A(agent.id,:) = assignments;
            vagents(v_i).selected_tasks = find(assignments);
            curr_utility = utility_calculation(vagents(v_i), P, Y);
            vagents(v_i).utility= curr_utility;
            curr_cost = -1*curr_utility;
            Cx(1,v_i) = curr_cost;
        end

        % compute the SUSD direction over the parameter space
        cov_x = cov(theta_');
        try
            [n,~] = eigs(cov_x,1,'SM');
        catch
            disp('error of Eigen decomposition');
        end
        % n_ = n*sign(n'*n_);
        if t == 1
            n_old = n;
        end 
        n_ = n*sign(n'*n_old);
        n_old = n_;
        [zmin, zarg] = min(Cx);
        
        % formation controller input
        d = theta_-mean(theta_,2);
        dnorm = vecnorm(d);
        umin = (d).*(d0-dnorm)./(dnorm.^2);
        umin(:,zarg) = 0; 
        
        % why set formation vector of the v-agent with
        %smallest measurement/cost to 0 ?? doesn't want the smallest
        %v-agent to move
        
        % apply exp mapping
        z = 1-exp(-(Cx-zmin)); % for avoiding vanishing or exploding gradients
        % z = Cx; % old approach 

        theta_ = theta_ + k1.*n_*z + k2.*umin;
%         theta_ = theta_./sum(theta_);
        % get the minimum parameters

        
        % save the best
        % curr_utility = evaluation_from_theta(vagents(zarg), P, Y, 'L_largest', 1);

        assignments = theta_to_assignment(vagents(zarg).theta, max_num_task_selected, 'L_largest');
        vagents(zarg).assignments = assignments;
        vagents(zarg).A(agent.id,:) = assignments;
        curr_utility = utility_calculation(vagents(zarg), P, Y);
        if curr_utility > utilty_best
            best_theta = vagents(zarg).theta;
            utilty_best = curr_utility;
        end


        % update the parameters in the graph
        for v_i=1:N_vagents
            if v_i ~= zarg
                if rand < eps
                    % randomize distribution
                    theta_(:,v_i) = rand(N_param, 1);
    %                 Theta_(:,v_i) = ones(N_param, 1)/N_param;
                else
                    % add guassian noise (normal distri)
                    theta_(:,v_i) = theta_(:,v_i)+0.0*randn(N_param,1);
                end 
            end 
        end
        % scaling 
        % theta_ = max(0, theta_);
        % if sum(theta_) ~= 0
        %     theta_ = theta_.*agent.max_num_tasks./sum(theta_);
        % end
        
        % projection (for randomized rounding)
        % theta_ = max(0, theta_);
        % theta_ = min(1, theta_);   
        

    

    end
    % scaling
    % theta_ = max(0, theta_);
    % if sum(theta_) ~= 0
    %     theta_ = theta_.*agent.max_num_tasks./sum(theta_);
    % end
    if exist('zarg','var') == 0
        pass
    end
    theta = theta_(:, zarg)';
    agent.theta = theta; 
    agent.Theta(agent_id,:) = agent.theta;
    assignments = theta_to_assignment(theta, max_num_task_selected, 'L_largest');
    agent.assignments = assignments;

    theta = zeros(size(best_theta));
    theta(1:N_tasks) = assignments*0.2;
    agent.theta = theta; 
    agent.Theta(agent_id,:) = agent.theta;

    agent.A(agent_id,:) = assignments;
    agent.selected_tasks = find(assignments);
    curr_utility = utility_calculation(agent, P, Y);
    agent.utility = curr_utility;
    x0 = agent.loc;
    [local_ordering, ~] = greedy_task_ordering(x0, P(:, agent.selected_tasks));
    agent.path = agent.selected_tasks(local_ordering);

    agent_self = agent;


end 