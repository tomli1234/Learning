#----------------------
# An R program to implement the Tic-Tac-Toe example in 
# Reinforcement Learning: An Introduction (Sutton and Barto)
#----------------------

rm(list=ls())

library(microbenchmark)
library(Rcpp)
sourceCpp('C:\\Users\\tomli\\Desktop\\myRcpp\\learn_progress_C.cpp')
sourceCpp('C:\\Users\\tomli\\Desktop\\myRcpp\\check_which_state_C.cpp')
sourceCpp('C:\\Users\\tomli\\Desktop\\myRcpp\\check_which_state_2_C.cpp')
sourceCpp('C:\\Users\\tomli\\Desktop\\myRcpp\\which_equal_C.cpp')

# check_which_state_C(matrix(rep(-1, 9), ncol=9), c(1, 1,-1,-1,-1,-1,-1,-1,-1))
# check_which_state_2_C(rbind(matrix(1, 8000,9),rep(2,9)), matrix(2, 3,9))

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

check_status <- function(state,
						 turn){
	state[state == -1] <- NA
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

cppFunction('double check_status_C(NumericVector x, int turn){
	NumericMatrix y, x_mat(3 ,3, x.begin());
	y = x_mat;
	// Check row
	for(int i=0; i<3; i++) {
		int consec = 0;
		for(int j=1; j<3; j++) {
			if(y(i, j) > -1){
				if(y(i, j) == y(i, j-1)) {
					consec += 1;
				}
				if(consec == 2) {
					return 1 - abs(y(i, j) - turn);
				}
			}
		}
	}
	// Check column
	y = transpose(x_mat);
	for(int i=0; i<3; i++) {
		int consec = 0;
		for(int j=1; j<3; j++) {
			if(y(i, j) > -1){
				if(y(i, j) == y(i, j-1)) {
					consec += 1;
				}
				if(consec == 2) {
					return 1 - abs(y(i, j) - turn);
				}
			}
		}
	}
	
	// Diagonal 1
	y = x_mat;
	NumericMatrix out1, out2;
	NumericVector out, list1;
	int nrow = y.nrow(), ncol = y.ncol();
	Function f1("row"),f2("col"); 
	out1 = f1(y);
	out2 = f2(y);
	out = out1 + out2;
	list1 = unique(out);
	for(int i = 0; i<list1.size(); i++) {
		int consec = 0;
		int x_prev = -2;
		for(int j =0; j <x.size(); j++) {
			if(out[j] == list1[i]) {
				if(x[j] == x_prev) {
					consec += 1;
				}		
				x_prev = x[j];
			}
			if(consec == 2) {
				return x[j];
			}
		}
	}

	// Diagonal 2	
	NumericMatrix revX = x_mat;
	ncol = x_mat.ncol();
	NumericVector x_i;
	for(int i=0; i<ncol; i++) {
		x_i = x_mat(_, i);
		std::reverse(x_i.begin(), x_i.end());
		revX(_ , i) = x_i;
	}
	
	y = revX;
	NumericVector x2;
	for(int i=0; i<3; i++) {
		for(int j=0; j<3; j++) {
			x2.push_back(y(i,j));
		}
	}
	x = x2;
	
	out1 = f1(y);
	out2 = f2(y);
	out = out1 + out2;
	list1 = unique(out);
	for(int i = 0; i<list1.size(); i++) {
		int consec = 0;
		int x_prev = -2;
		for(int j =0; j <x.size(); j++) {
			if(out[j] == list1[i]) {
				if(x[j] == x_prev) {
					consec += 1;
				}		
				x_prev = x[j];
			}
			if(consec == 2) {
				return x[j];
			}
		}
	}	

	
	return -1;
}')
current_state <- c(0, -1, -1, -1, 0, -1, -1, -1, 0)
check_status_C(current_state, 0)

microbenchmark(
check_status(current_state, 0),
check_status_C(current_state, 0)
)


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
	backup_state <- list(matrix(-2, ncol = 9),matrix(-2, ncol = 9))
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
		random_move <- runif(1) < random
		if(random_move){
			which_option <- sample(option, 1)
		} else {
			which_option <- option[sample.vec(which_equal_C(decision_values, max(decision_values)), 1)]
		}
		decision <- learned_state[[1 + turn]][which_option, ]
		last_move <- check_which_state_2_C(as.matrix(learned_state[[1 + turn]][, 1:9]), matrix(backup_state[[1 + turn]], ncol = 9))			
		old_value <- learned_state[[1 + turn]][last_move, 10]
		current_state <- decision[1:9]
		current_status <- check_status(current_state, turn)
		
		## Learning---------------------
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
		oppo_state <- check_which_state_2_C(as.matrix(learned_state[[1 + turn]][, 1:9]), matrix(backup_state[[1 + turn]], ncol = 9))					
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
	state[state == -1] <- NA
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
	current_state[current_state == -1] <- NA
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
	current_state <- rep(-1,9)
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