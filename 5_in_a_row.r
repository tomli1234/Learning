#----------------------
# An R program to implement the Tic-Tac-Toe example in 
# Reinforcement Learning: An Introduction (Sutton and Barto)
#----------------------

rm(list=ls())

library(microbenchmark)
library(Rcpp)
library(TTTrcpp)

possible_move <- function(current_state,
						  turn){
	possible_decision <- which_equal_C(current_state, -1)
	sapply(possible_decision, function(x) {
			current_state[x] <- turn
			return(state = current_state)	
		}
	)
}

sample.vec <- function(x, ...) x[sample(length(x), ...)]

## Initialisation
learning <- function(alpha = 0.1, random = 0.1,
					 rounds,
					 learned_state = NULL) {
	if(is.null(learned_state)) {
		learned_state <- NULL
		learned_state[[1]] <- matrix(c(rep(-1, 25), 0.5, 0), 1, 27)
		learned_state[[2]] <- matrix(c(rep(-1, 25), 0.5, 0), 1, 27)
	}
	progress <- NULL

	## Learning
	for(i in 1:rounds){
		# alpha <- 1/i^(1/2.5)
		current_state <- rep(-1,25)
		turn <- sample(0:1, 1)
		which_option_hist <- list(NULL,NULL)
		while(check_status_C(current_state, turn) == -1){
		
			## Update experience--------------
			x <- t(possible_move(current_state, turn = turn))
			x_base3 <- apply(x + 1, 1, base3_to_decimal)
			# microbenchmark(
			# learned	<- check_which_state_2_C(as.matrix(learned_state[[1 + turn]][, 1:25]), x),
			# learned <- sapply(x_base3, function(x) which_equal_C(learned_state[[1 + turn]][,27], x)),
			learned <- which_equal_C_2(learned_state[[1 + turn]][,27], x_base3)
			# )
			
			option <- learned[learned!=0]
			# If not seen possible move, then assign it with 0.5
			if(sum(learned == 0) > 0){
				option <- c(option, nrow(learned_state[[1 + turn]]) + 1:sum(learned == 0))
				new_state <- x[learned == 0, ]
				learned_state[[1 + turn]] <- rbind(learned_state[[1 + turn]], 
												cbind(matrix(new_state, nrow=sum(learned == 0)), 
													  0.5,
													  base3_to_decimal(new_state + 1)))
			}
				
			## Decision----------------------	
			decision_values <- learned_state[[1 + turn]][option, 26]
			random_move <- runif(1) < random
			if(random_move){
				which_option <- sample(option, 1)
			} else {
				which_option <- option[sample.vec(which_equal_C(decision_values, max(decision_values)), 1)]
			}
			which_option_hist[[1 + turn]] <- c(which_option_hist[[1 + turn]], which_option)
			decision <- learned_state[[1 + turn]][which_option, ]
			last_move <- which_option_hist[[1 + turn]][length(which_option_hist[[1 + turn]]) - 1]
			
			old_value <- learned_state[[1 + turn]][last_move, 26]
			current_state <- decision[1:25]
			current_status <- check_status_C(current_state, turn)
			
			## Learning---------------------
			### Current move
			if(current_status == -1){
				new_value <- decision[26]
				learned_state[[1 + turn]][last_move, 26] <- old_value + alpha * (new_value - old_value)
			} else {
				new_value <- current_status
				learned_state[[1 + turn]][last_move, 26] <- old_value + alpha * (new_value - old_value)
				learned_state[[1 + turn]][which_option, 26] <- new_value
			}
						
			turn <- abs(turn - 1)
			
			### Learning from opponent's move (learning defensive move)
			oppo_state <- which_option_hist[[1 + turn]][length(which_option_hist[[1 + turn]])]
			oppo_value <- learned_state[[1 + turn]][oppo_state, 26]
			oppo_status <- check_status_C(current_state, turn)
			if(oppo_status == -1){
				# new_value <- decision[10]
				# learned_state[[1 + turn]][oppo_state, 10] <- oppo_value + alpha * (new_value - oppo_value)
			} else {
				new_value <- oppo_status
				learned_state[[1 + turn]][oppo_state, 26] <- oppo_value + alpha * (new_value - oppo_value)
				# print(learned_state[[1 + turn]][oppo_state, 10] )
			}	
					
		}
		print(paste0(i,', ', nrow(learned_state[[1]])))
		progress <- c(progress, learn_progress_C(learned_state[[1]][,26]))
		plot(progress, type='l')
	}
	return(learned_state)
}

# Parallel learning (Shadow clone learning)
library(parallel)

shadow_clone <- function(learner_num, sub_rounds) {
	envir_1 <- environment()
	no_cores <- detectCores() - 1
	cl <- makeCluster(no_cores)		
	clusterExport(cl, list("learning","possible_move","sample.vec","which_equal_C","which_equal_C_2","check_status_C"),
					envir = .GlobalEnv)
	learners <- lapply(1:learner_num, function(x) NULL)
	for(j in 1:10) {

	clusterExport(cl, list("learners"), envir = envir_1) # set envir as "inside the function" since "learners" not defined in global
		learners <- parLapply(cl, 
						1:learner_num, 
						function(x) {
						learning(rounds = sub_rounds,
								 learned_state = learners[[x]])	
					}
				)	
		
		# Combine learners
		learned_state_all <- NULL
		k <- 2
		while(k <= learner_num) {
			for(i in 1:2) {
				same <- check_which_state_2_C(as.matrix(learners[[k-1]][[i]][, -26]), as.matrix(learners[[k]][[i]][, -26]))								
				learned_1 <- learners[[k-1]][[i]][same, 26]
				learned_2 <- learners[[k]][[i]][which(same > 0), 26]
				learners[[k-1]][[i]][same, 26] <- apply(cbind(learned_1,learned_2), 1, mean)
				disjoint <-	which_equal_C(same, 0)		
				learned_state_all[[i]] <- rbind(learners[[k-1]][[i]], learners[[k]][[i]][disjoint,])	
			}
			learners <- lapply(1:learner_num, function(x) learned_state_all)
			k <- k + 1
		}	
	}
	stopCluster(cl)
	return(learners)
}

microbenchmark(
# learners <- shadow_clone(learner_num = 3, sub_rounds = 100),
learner_2 <- learning(rounds = 300, learned_state = NULL),
times = 1)


check_finish <- function(state){
	state[state == -1] <- NA
	state_mat <- matrix(state, 5, 5)
	colsum1 <- colSums(state_mat)
	rowsum1 <- rowSums(state_mat)
	diagsum1 <- c(sum(diag(state_mat)), sum(diag(state_mat[nrow(state_mat):1, ])))
	if(any(c(colsum1, rowsum1, diagsum1) == 5, na.rm=TRUE)){
		return('Player 1 wins') # Win
	} else if(any(c(colsum1, rowsum1, diagsum1) == 0, na.rm=TRUE)){
		return('Player 0 wins') # Lose
	} else if(any(is.na(state), na.rm=TRUE) == FALSE){
		return("It's a draw") # Draw
	} else {
		return(0)
	}
}

# Visualise game
library(ggplot2)

blank_theme <- theme_minimal()+
      theme(
            axis.title = element_blank(),
            axis.text = element_blank(),
            panel.border = element_blank(),
            panel.grid=element_blank(),
            axis.ticks = element_blank(),
            plot.title=element_text(size=14, face="bold")
      )
visualise_game <- function(current_state){	
	current_state[current_state == -1] <- NA
	visual_data <- data.frame(expand.grid(x = 1:5, y = 1:5), current_state)
	visual_data$current_state <- ifelse(visual_data$current_state == 1, 'O', 'X')
	ggplot(visual_data, aes(x = x, y = y)) +
		geom_vline(xintercept = seq(0.5, 3.5, 1))+
		geom_hline(yintercept = seq(0.5, 3.5, 1))+
		scale_x_continuous(limit = c(0.5,3.5), expand = c(0,0))+
		scale_y_continuous(limit = c(0.5,3.5), expand = c(0,0))+
		theme_classic()+
		blank_theme+
		geom_text(aes(x = x , y = y, label = current_state), size = 24)+
		ggtitle('O: me\nX: machine')
}		
# visualise_game(current_state)
	

play <- function(first){
	current_state <- rep(-1,25)
	turn = 0
	while(check_finish(current_state) == 0){
		if(first == 1){
			print(matrix(current_state, 5, 5))
			# print(visualise_game(current_state))
			player_move <- readline('Please select a move\n')
			current_state[as.numeric(player_move)] <- 1
			if(check_finish(current_state) != 0) {break}
		}	
		x <- t(possible_move(current_state, turn = turn))
		learned <- split(x, row(x)) %in% split(learned_state[[1 + turn]][, 1:25], matrix(rep(1:nrow(learned_state[[1 + turn]]), each = 25), nrow = nrow(learned_state[[1 + turn]]), byrow = TRUE))
		if(sum(learned == FALSE) != 0){
			learned_state[[1 + turn]] <- rbind(learned_state[[1 + turn]], cbind(matrix(x[learned == FALSE, ], nrow=sum(learned == FALSE)), 0.5))
		}
		option <- which(split(learned_state[[1 + turn]][, 1:25], matrix(rep(1:nrow(learned_state[[1 + turn]]), each = 25), nrow = nrow(learned_state[[1 + turn]]), byrow = TRUE)) %in% split(x, row(x)) )
		decision_values <- learned_state[[1 + turn]][option, 26]
		which_option <- option[sample.vec(which(decision_values == max(decision_values)), 1)]
		decision <- learned_state[[1 + turn]][which_option, ]
		current_state <- decision[1:25]
		if(first == 0){
			print(matrix(current_state, 5, 5))
			# print(visualise_game(current_state))
			player_move <- readline('Please select a move\n')
			current_state[as.numeric(player_move)] <- 1
		}
	}		
	# g <- visualise_game(current_state)
	# if(check_finish(current_state) == 'Player 1 wins'){
		# g <- g + annotate('text', x = 2, y = 2, label = 'O wins', size = 20, colour = 'skyblue2')
	# } else if(check_finish(current_state) == 'Player 0 wins'){
		# g <- g + annotate('text', x = 2, y = 2, label = 'X wins', size = 20, colour = 'skyblue2')		
	# } else {
		# g <- g + annotate('text', x = 2, y = 2, label = 'Draw', size = 20, colour = 'skyblue2')				
	# }
	# print(g)
	check_finish(current_state)
}
learned_state <- learners[[1]]
learned_state <- learner_2
play(first=0)



# Test 
test_play <- function(first, player1, player0){
	current_state <- rep(-1,9)
	learned_state <- NULL
	learned_state[[1]] <- player0
	learned_state[[2]] <- player1
	
	turn = first
		while(is.null(check_status(current_state, turn))){
		
			## Update experience--------------
			x <- t(possible_move(current_state, turn = turn))
			learned	<- check_which_state_2_C(as.matrix(learned_state[[1 + turn]][, 1:9]), x)
			# If not seen possible move, then assign it with 0.5
			if(sum(learned == 0) > 0){
				learned_state[[1 + turn]] <- rbind(learned_state[[1 + turn]], 
												cbind(matrix(x[learned == 0, ], 
														nrow=sum(learned == 0)), 0.5))
			}
				
			## Decision----------------------
			option	<- check_which_state_2_C(as.matrix(learned_state[[1 + turn]][, 1:9]), x)			
			decision_values <- learned_state[[1 + turn]][option, 10]
			which_option <- option[sample.vec(which_equal_C(decision_values, max(decision_values)), 1)]
			decision <- learned_state[[1 + turn]][which_option, ]
			current_state <- decision[1:9]
			# current_status <- check_status(current_state, turn)
			turn <- abs(turn - 1)
		}	
	return(check_finish(current_state))
}

test_play_loop <- function(rounds = 1000, player1, player0) {
	result <- NULL
	for(i in 1:rounds) {
		first <- sample(0:1, 1)
		result_i <- test_play(first = first, 
					player1 = player1, 
					player0 = player0)
		result <- c(result, result_i)
	}
	table(result)
}
test_play_loop(rounds = 1000, 
				player1 = learner_2[[2]], 
				player0 = learners[[1]][[1]])


library(animation)
# Animation
saveGIF(for(i in 0:5){
			first <- i %% 2
			play(first = first)
		},
	interval = 2, 
	movie.name="C:\\Users\\tomli\\Desktop\\tic_tac_toe.gif")


current_state <- c(-1, -1, -1, -1, -1, -1, -1, -1, -1)
turn = 0

decision

late_game <- apply(learned_state[[1 + turn]][,1:9], 1, function(x) sum(is.na(x))) == 6
plot(table(learned_state[[1 + turn]][late_game,10]))
plot(table(learned_state[[1 + turn]][,10]))

learned_state[[1 + turn]][late_game,]