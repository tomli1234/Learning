#----------------------
# An R program to implement the Tic-Tac-Toe example in 
# Reinforcement Learning: An Introduction (Sutton and Barto)
#----------------------

rm(list=ls())

library(microbenchmark)
library(Rcpp)

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

cppFunction('double learn_progress_C(NumericVector x){
	int n = x.size();
	double total = 0;
	for(int i; i < n; ++i) {
		total += pow(x[i] - 0.5, 2.0);
	}
	return total/n;
}')
# learn_progress <- function(learned_state){
	# mean((learned_state[,10] - 0.5)^2)
# }

check_which_state <- function(learned_state, current_state){
	i <- 0
	matched <- FALSE
	while(!matched){
		i <- i + 1
		matched <- sum(learned_state[i,][1:9] - current_state) == 0
	}
	return(i)
}
check_which_state(learned_state[[1 + turn]], current_state)

## Initialisation
alpha <- 0.1
random <- 0.1
learned_state <- NULL
learned_state[[1]] <- matrix(c(rep(-1, 9), 0.5), 1, 10)
learned_state[[2]] <- matrix(c(rep(-1, 9), 0.5), 1, 10)
progress <- NULL

## Learning
for(i in 1:60000){
	# alpha <- 1/i^(1/2.5)
	current_state <- rep(-1,9)
	turn <- sample(0:1, 1)
	backup_state <- list(NA,NA)
	while(is.null(check_status(current_state, turn))){
	
		## Update experience
		learned <- list(current_state) %in% 
					split(learned_state[[1 + turn]][, 1:9], 
							matrix(rep(1:nrow(learned_state[[1 + turn]]), each = 9), 
								nrow = nrow(learned_state[[1 + turn]]), 
								byrow = TRUE))
learned_state[[1 + turn]][, 1:9] - current_state

		if(learned == FALSE){
			learned_state[[1 + turn]] <- rbind(learned_state[[1 + turn]], 
												c(current_state, 0.5))
		}
		x <- t(possible_move(current_state, turn = turn))
		learned <- split(x, row(x)) %in% 
					split(learned_state[[1 + turn]][, 1:9], 
						matrix(rep(1:nrow(learned_state[[1 + turn]]), each = 9), 
							nrow = nrow(learned_state[[1 + turn]]), 
							byrow = TRUE))
		if(sum(learned == FALSE) != 0){
			learned_state[[1 + turn]] <- rbind(learned_state[[1 + turn]], 
											cbind(matrix(x[learned == FALSE, ], 
													nrow=sum(learned == FALSE)), 0.5))
		}
			
		## Decision
		option <- which(split(learned_state[[1 + turn]][, 1:9], 
								matrix(rep(1:nrow(learned_state[[1 + turn]]), each = 9), 
									nrow = nrow(learned_state[[1 + turn]]), 
									byrow = TRUE)) 
						%in% split(x, row(x)))
		decision_values <- learned_state[[1 + turn]][option, 10]
		random_move <- runif(1) < random
		if(random_move){
			which_option <- sample(option, 1)
		} else {
			which_option <- option[sample.vec(which(decision_values == max(decision_values)), 1)]
		}
		decision <- learned_state[[1 + turn]][which_option, ]
		last_move <- which(split(learned_state[[1 + turn]][, 1:9], 
									matrix(rep(1:nrow(learned_state[[1 + turn]]), each = 9), 
										nrow = nrow(learned_state[[1 + turn]]), 
										byrow = TRUE)) 
								%in% list(backup_state[[1 + turn]]))
		old_value <- learned_state[[1 + turn]][last_move, 10]
		current_state <- decision[1:9]
		current_status <- check_status(current_state, turn)
		
		## Learning
		### Current move
		if(is.null(current_status)){
			new_value <- decision[10]
			learned_state[[1 + turn]][last_move, 10] <- old_value + alpha * (new_value - old_value)
		} else {
			new_value <- current_status
			learned_state[[1 + turn]][last_move, 10] <- old_value + alpha * (new_value - old_value)
			learned_state[[1 + turn]][which_option, 10] <- new_value
		}
		
		backup_state[[1 + turn]] <- current_state
		
		turn <- abs(turn - 1)
		
		### Learning from opponent's move (learning defensive move)
		oppo_state <- which(split(learned_state[[1 + turn]][, 1:9], 
									matrix(rep(1:nrow(learned_state[[1 + turn]]), each = 9), 
											nrow = nrow(learned_state[[1 + turn]]), 
											byrow = TRUE)) 
									%in% list(backup_state[[1 + turn]]))
		oppo_value <- learned_state[[1 + turn]][oppo_state, 10]
		oppo_status <- check_status(current_state, turn)
		if(is.null(oppo_status)){
			# new_value <- decision[10]
			# learned_state[[1 + turn]][oppo_state, 10] <- oppo_value + alpha * (new_value - oppo_value)
		} else {
			new_value <- oppo_status
			learned_state[[1 + turn]][oppo_state, 10] <- oppo_value + 0.5 * (new_value - oppo_value)
			print(learned_state[[1 + turn]][oppo_state, 10] )
		}	
				
	}
	print(paste0(i,', ', nrow(learned_state[[1]])))
	progress <- c(progress, learn_progress_C(learned_state[[1]][,10]))
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
	visual_data <- data.frame(expand.grid(x = 1:3, y = 1:3), current_state)
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
	current_state <- rep(NA,9)
	turn = 0
	while(check_finish(current_state) == 0){
		if(first == 1){
			print(matrix(current_state, 3, 3))
			print(visualise_game(current_state))
			player_move <- readline('Please select a move\n')
			current_state[as.numeric(player_move)] <- 1
			if(check_finish(current_state) != 0) {break}
		}	
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
		if(first == 0){
			print(matrix(current_state, 3, 3))
			print(visualise_game(current_state))
			player_move <- readline('Please select a move\n')
			current_state[as.numeric(player_move)] <- 1
		}
	}		
	g <- visualise_game(current_state)
	if(check_finish(current_state) == 'Player 1 wins'){
		g <- g + annotate('text', x = 2, y = 2, label = 'O wins', size = 20, colour = 'skyblue2')
	} else if(check_finish(current_state) == 'Player 0 wins'){
		g <- g + annotate('text', x = 2, y = 2, label = 'X wins', size = 20, colour = 'skyblue2')		
	} else {
		g <- g + annotate('text', x = 2, y = 2, label = 'Draw', size = 20, colour = 'skyblue2')				
	}
	print(g)
	check_finish(current_state)
}
play(first=1)

library(animation)
# Animation
saveGIF(for(i in 0:5){
			first <- i %% 2
			play(first = first)
		},
	interval = 2, 
	movie.name="C:\\Users\\tomli\\Desktop\\tic_tac_toe.gif")


current_state <- c(0, NA, 1, NA, NA, NA, NA, NA, 1)
turn = 0

decision

late_game <- apply(learned_state[[1 + turn]][,1:9], 1, function(x) sum(is.na(x))) == 6
plot(table(learned_state[[1 + turn]][late_game,10]))
plot(table(learned_state[[1 + turn]][,10]))

learned_state[[1 + turn]][late_game,]