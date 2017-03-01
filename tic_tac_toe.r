#----------------------
# An R program to implement the Tic-Tac-Toe example in 
# Reinforcement Learning: An Introduction (Sutton and Barto)
#----------------------

rm(list=ls())

possible_move <- function(current_state,
						  turn){
	possible_decision <- which(is.na(current_state))
	sapply(possible_decision, function(x) {
			current_state[x] <- turn
			return(state = current_state)	
		}
	)
}

sample.vec <- function(x, ...) x[sample(length(x), ...)]

check_status <- function(state,
						 turn){
	state_mat <- matrix(state, 3, 3)
	colsum1 <- colSums(state_mat)
	rowsum1 <- rowSums(state_mat)
	diagsum1 <- c(sum(diag(state_mat)), sum(diag(state_mat[nrow(state_mat):1, ])))
	if(any(c(colsum1, rowsum1, diagsum1) == 3, na.rm=TRUE)){
		return(turn)  # Player 1 Wins
	} else if(any(c(colsum1, rowsum1, diagsum1) == 0, na.rm=TRUE)){
		return(1 - turn) # Player 1 Loses
	} else if(any(is.na(state), na.rm=TRUE) == FALSE){
		return(1 - turn) # Draw
	}
}

learn_progress <- function(learned_state){
	mean((learned_state[,10] - 0.5)^2)
}

## Initialisation
alpha <- 0.5
random <- 0.1
learned_state <- NULL
learned_state[[1]] <- matrix(c(rep(NA, 9), 0.5), 1, 10)
learned_state[[2]] <- matrix(c(rep(NA, 9), 0.5), 1, 10)
progress <- NULL

## Learning
for(i in 1:30000){
	current_state <- rep(NA,9)
	turn <- sample(0:1, 1)
	while(is.null(check_status(current_state, turn))){
		## Update experience
		learned <- list(current_state) %in% split(learned_state[[1 + turn]][, 1:9], matrix(rep(1:nrow(learned_state[[1 + turn]]), each = 9), nrow = nrow(learned_state[[1 + turn]]), byrow = TRUE))
		if(learned == FALSE){
			learned_state[[1 + turn]] <- rbind(learned_state[[1 + turn]], c(current_state, 0.5))
		}
		x <- t(possible_move(current_state, turn = turn))
		learned <- split(x, row(x)) %in% split(learned_state[[1 + turn]][, 1:9], matrix(rep(1:nrow(learned_state[[1 + turn]]), each = 9), nrow = nrow(learned_state[[1 + turn]]), byrow = TRUE))
		if(sum(learned == FALSE) != 0){
			learned_state[[1 + turn]] <- rbind(learned_state[[1 + turn]], cbind(matrix(x[learned == FALSE, ], nrow=sum(learned == FALSE)), 0.5))
		}
			
		## Decision
		option <- which(split(learned_state[[1 + turn]][, 1:9], matrix(rep(1:nrow(learned_state[[1 + turn]]), each = 9), nrow = nrow(learned_state[[1 + turn]]), byrow = TRUE)) %in% split(x, row(x)) )
		decision_values <- learned_state[[1 + turn]][option, 10]
		random_move <- runif(1) < random
		if(random_move){
			which_option <- sample(option, 1)
		} else {
			which_option <- option[sample.vec(which(decision_values == max(decision_values)), 1)]
		}
		decision <- learned_state[[1 + turn]][which_option, ]
		last_state <- which(split(learned_state[[1 + turn]][, 1:9], matrix(rep(1:nrow(learned_state[[1 + turn]]), each = 9), nrow = nrow(learned_state[[1 + turn]]), byrow = TRUE)) %in% list(current_state))
		old_value <- learned_state[[1 + turn]][last_state, 10]
		current_state <- decision[1:9]
		current_status <- check_status(current_state, turn)
		## Learning
		if(is.null(current_status)){
			new_value <- decision[10]
			learned_state[[1 + turn]][last_state, 10] <- old_value + alpha * (new_value - old_value)
		} else {
			new_value <- current_status
			learned_state[[1 + turn]][last_state, 10] <- old_value + alpha * (new_value - old_value)
			learned_state[[1 + turn]][which_option, 10] <- new_value
		}
		turn <- abs(turn - 1)
	}
	print(paste0(i,', ', nrow(learned_state[[1]])))
	progress <- c(progress, learn_progress(learned_state[[1]]))
	plot(progress, type='l')
}


check_finish <- function(state){
	state_mat <- matrix(state, 3, 3)
	colsum1 <- colSums(state_mat)
	rowsum1 <- rowSums(state_mat)
	diagsum1 <- c(sum(diag(state_mat)), sum(diag(state_mat[nrow(state_mat):1, ])))
	if(any(c(colsum1, rowsum1, diagsum1) == 3, na.rm=TRUE)){
		return('Player 1 wins') # Win
	} else if(any(c(colsum1, rowsum1, diagsum1) == 0, na.rm=TRUE)){
		return('Player 0 wins') # Lose
	} else if(any(is.na(state), na.rm=TRUE) == FALSE){
		return("It's a draw") # Draw
	} else {
		return(0)
	}
}

play <- function(){
	current_state <- rep(NA,9)
	turn = 0
	while(check_finish(current_state) == 0){
		# print(matrix(current_state, 3, 3))
		# player_move <- readline('Please select a move\n')
		# current_state[as.numeric(player_move)] <- 1
		x <- t(possible_move(current_state, turn = turn))
		learned <- split(x, row(x)) %in% split(learned_state[[1 + turn]][, 1:9], matrix(rep(1:nrow(learned_state[[1 + turn]]), each = 9), nrow = nrow(learned_state[[1 + turn]]), byrow = TRUE))
		if(sum(learned == FALSE) != 0){
			learned_state[[1 + turn]] <- rbind(learned_state[[1 + turn]], cbind(matrix(x[learned == FALSE, ], nrow=sum(learned == FALSE)), 0.5))
		}
		option <- which(split(learned_state[[1 + turn]][, 1:9], matrix(rep(1:nrow(learned_state[[1 + turn]]), each = 9), nrow = nrow(learned_state[[1 + turn]]), byrow = TRUE)) %in% split(x, row(x)) )
		decision_values <- learned_state[[1 + turn]][option, 10]
		which_option <- option[sample.vec(which(decision_values == max(decision_values)), 1)]
		decision <- learned_state[[1 + turn]][which_option, ]
		current_state <- decision[1:9]
		print(matrix(current_state, 3, 3))
		player_move <- readline('Please select a move\n')
		current_state[as.numeric(player_move)] <- 1
	}		
	print(matrix(current_state, 3, 3))	
	check_finish(current_state)
}
play()

current_state <- c(NA, NA, NA, NA, 1, NA, 1, 0, 0)
turn = 0

decision




