## Imports 
from tqdm.notebook import tqdm
import numpy as np

## Ignore annoying warning
import warnings
warnings.filterwarnings("ignore")

class genetic_algorithm:
    '''
    implements the feature selection using a genetic algorithm. Goal is to reduce the number of features to use for the latter machine learning model to reduce processing time and capacity
    '''
    
    def __init__(self, num_populations:int = 10, num_parents:int=2) -> None:
        '''
        constructor of class
        Parameters:
            - num_populations: desired number of populations [Integer, defaukt = 10]
            - num_parents: desired number of parents to use for crossover combination [Integer, default = 2]
        Initializes:
            - num_populations
            - num_parents
            - best_fit: List containing the best fits for the problem [List]
        Returns:
            - None
        '''
        self.num_populations = num_populations
        self.num_parents = num_parents
        self.best_fit = []
    
    def build_population(self, data:np.array) -> np.array:
        '''
        function that builds up a evenly distributed (in respect of the labels) dataset with reduced number of features to work with
        Parameters:
            - data: original data of each population [numpy.array]
        Returns:
            - population: population for this generation [numpy.array]
        '''
        population = np.copy(data)
        return population
    
    def mix_data(self, X:np.array) -> np.array:
        '''
        function to mix up the data equally. Means in the end, the class contains a list of DataFrames that are all equally distributed (concerning the labels)
        Parameters:
            - X: whole x-values data [numpy.array]
        Returns:
            - self.X: reduced dataseet of x [numpy.array]
        '''
        ## init the number of populations as ones and the rest of remaining indize as zeros
        ones = np.ones(self.num_populations)
        zeros = np.zeros(X.shape[0] - ones.__len__())
        indize = np.append(ones,zeros)
        ## shuffle
        np.random.shuffle(indize)
        ## set the list of numpy.array's
        self.X = np.array([d for i,d in enumerate(X) if indize[i]==1])
        return self.X
            
    def drop_worst(self, X:np.array, costs:np.array) -> np.array:
        '''
        function used to drop the worst 10% of DataFrames per population - not used right now in 'main()'
        Parameters:
            - X: x-values data [numpy.array]
            - costs: calculated costs of each population [numpy.array]
        Returns:
            - X: x-values data without worst 10% [numpy.array]
        '''
        ## calc number of populations to drop
        num_drops = int(self.num_populations / 10)
        ## get the 'worst' 10% of the DataFrames
        indize = np.where(np.in1d(costs, sorted(costs)[-num_drops:]))[0].tolist()
        ## keep all subdatasets that are not in those indize, return X
        X = [d for idx,d in enumerate(X) if idx not in indize]
        return X
    
    def cost(self, X:np.array, y:np.array) -> np.array:
        '''
        function that calculates the cost that each population produces
        Parameters:
            - X: x-values data [numpy.array]
            - y: y-values data [numpy.array]
        Returns:
            - costs: array with all costs that all populations produced in respective generation [numpy.array]
        '''
        ## define costs of each 'evaluation metric'
        cost_overbooked = 1
        cost_underbooked = 1
        ## init the costs
        costs = []
        ## calculate how many people are needed
        needed_booking = y
        for x in self.X:
            ## get the respective population
            population = self.build_population(x)
            ## calculate how many people are available
            got_booking = population
            difference = (got_booking-needed_booking).reshape(y.shape[1],y.shape[0])
            ## calculate the cost of each hour on each day
            cost = []
            for d in difference:
                c = 0
                for diff in d:
                    ## if more people available than needed
                    if diff >= 0:
                        c+=diff*cost_overbooked
                    ## otherwise
                    else:
                        c-=diff*cost_underbooked
                cost.append(c)
            costs.append(np.array(cost))            
        return np.array(costs)
    
    def select_mating_pool(self, costs:np.array) -> np.array:
        '''
        function to select the best individuals in the current generation as parents for producing the offspring of the next generation
        Parameters:
            - costs: array with all costs that all populations produced in respective generation [numpy.array]
        Returns:
            - parents: the best 'num_parents' individuals in the current x-values generation [numpy.array]
        '''
        ## init the parents
        parents = []
        ## search as many parents as 'num_parents' intends to do
        for i in range(self.num_parents):
            ## get the one with the lowest costs, set their cost to 'a million' that it won't be chosen multiple times
            idx = np.argmin([np.min(np.sum(cost)) for cost in costs])
            parents.append(self.X[idx])
            ## add the best fit of this generation to best_fit
            if i==0:
                self.best_fit.append(self.X[idx])
            ## reset the costs of that index that it shall not be used twice
            costs[idx] = 99999999
        return np.array(parents)
    
    def crossover(self, datalist:np.array) -> np.array:
        '''
        function that performs the crossover of the 'num_parents' best individuals, also called 'parents'
        Parameters:
            - datalist: List containing the best 'num_parents' individuals - the parents [numpy.array]
        Returns:
            - offspring: List containing mixed numpy.arrays built up from the parents - number of different 'children' is num_populations - num_parents [numpy.array]
        '''
        ## init the offspring
        offspring = []
        for _ in range(self.num_populations - self.num_parents):
            ## get an array that indicates which feature will be used from which parent (randomly) by the shape of hours and days -> availability of employees are randomly chosen per day and hour
            parents_array = np.random.choice(range(self.num_parents), (datalist.shape[-2],datalist.shape[-1]))
            ## for each parent: get the array with 0s and 1s that shows which features of this respective parent will be used
            parents = [ np.logical_not(parents_array!=i).astype(int) for i in range(self.num_parents) ]
            ## init the list of DataFrames for this respective 'children'
            crossover = np.array([])
            ## go through all parents
            for i,data in enumerate(datalist):
                ## get the chromosoms of the respective parent that will be used for this 'children'
                idx = np.where(parents[i].flatten() == 1)
                p = data.flatten()[idx]
                crossover = np.append(crossover, p)
            offspring.append(crossover.reshape(datalist.shape[-2], -1))
        return np.array(offspring)
        
    def mutation(self, datalist:np.array) -> np.array:
        '''
        function that perform the mutation of the offspring-'children'
        Parameters:
            - datalist: List containing the 'children' [numpy.array]
        Returns:
            - offspring_mutation: List of the mutated 'children' [numpy.array]
        '''
        def mutate(d:np.array, idx:np.array) -> np.array:
            '''
            actual mutation function
            Parameters:
                - d: chromosoms in childrens DataFrame [numpy.array]
                - idx: idx of chromosoms that mutate [numpy.array]
            Returns:
                - mutation: mutated row in childrens DataFrame [numpy.array]
            '''
            ## respective mutation - in this case a bitswap
            return np.logical_not(d[idx]).astype(int)
        ## calculate the number of mutations per children - in this case 20% of all genes
        num_mutations = int(datalist[0].flatten().shape[0] * 0.2)
        ## init the mutated 'children'
        offspring_mutation = []
        ## get the shape - to reshape later
        shape = datalist[0].shape
        ## go through all given 'children'
        for offspring_crossover in datalist:
            offspring_crossover = offspring_crossover.flatten()
            ## randomly choose 10% of the columns of the 'children'
            mutation_idx = np.random.choice(range(offspring_crossover.shape[0]), num_mutations)
            ## mutation changes the value of those columns
            offspring_crossover[mutation_idx] = mutate(offspring_crossover, mutation_idx)
            offspring_mutation.append(offspring_crossover.reshape(shape))
        return np.array(offspring_mutation)
    
    def train(self, X:np.array, y:np.array, generations:int, mutate:bool = True) -> np.array:
        '''
        training function that performs all steps for the genetic algorithm throughout all generations
        Parameters:
            - X: whole x-data [numpy.array]
            - y: whole y-data [numpy.array]
            - generations: number of generations to perform [Integer]
            - mutate: whether (=True) or not (=False) to make a mutation for the children [boolean, default = True]
        Returns:
            - best_fit: population with the best fit [numpy.array]
        '''
        ## mix the data
        population = self.mix_data(X)
        ## init the best outputs and the best features
        best_outputs = []
        best_fit = []
        ## init the bar to show the progress
        pbar = tqdm(total = generations)
        ## go through all generations
        for generation in range(generations):
            ## calculate the costs for this population, append their costs to best_outputs
            costs = self.cost(population, y)   
            best_outputs.append(np.min(np.sum(costs,axis=1))) 
            ## get the parents
            parents = self.select_mating_pool(costs)
            ## get the offspring = 'children'
            offspring_crossover = self.crossover(parents)
            ## mutate (or not) the 'children'
            if mutate:
                offspring_mutation = self.mutation(offspring_crossover)         
            else:
                offspring_mutation = offspring_crossover.copy()
            ## replace old data with the new 'children' and parents
            population[:self.num_parents] = parents
            population[self.num_parents:] = offspring_mutation
            ## update progress of training
            pbar.set_description(f'Epoch: {generation}; min costs: {best_outputs[-1]:.2f} / best \'til now: {np.min(best_outputs):.2f}')
            pbar.update(1)
        print(f'the best generation reaches a minimum cost of {np.min(best_outputs)}')
        ## get the generation with the lowest maximum of costs
        best_fit = self.best_fit[np.argmin([np.abs(bf-y).sum() for bf in self.best_fit])]
        return best_fit