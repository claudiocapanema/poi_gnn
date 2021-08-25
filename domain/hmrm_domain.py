import numpy as np
import pandas as pd
from numpy.linalg import norm
from numpy.linalg import inv as inverse
import scipy.sparse as sparse

class HmrmDomain:

    def __init__(self):
        self._user_location_frequency = np.array([])
        self._user_time_frequency = np.array([])
        self._location_co_ocurrency = np.array([])
        self._location_time = np.array([])
        self._weight = 0.001
        self.activity_location = np.array([])
        self.activity_time = np.array([])
        self.user_activity = np.array([])
        self.activity_embedding = np.array([])
        self.target_Location_embedding = np.array([])
        self.context_location_embedding = np.array([])
        self.time_slot_embedding = np.array([])
    
    def _create_user_location_frequency_matrix(self, users_checkins):
        
        placeids_int = users_checkins["placeid_int"].tolist()
        userids = users_checkins["userid_int"]
        total_users = len(users_checkins["userid_int"].unique())
        total_places = len(users_checkins["placeid_int"].unique())

        print("Shape: ", users_checkins.shape)
        self._user_location_frequency = sparse.lil_matrix((total_users, total_places))
        
        for i in range(len(placeids_int)):
            self._user_location_frequency[userids[i], placeids_int[i]] += 1

    def _create_user_time_frequency_matrix(self, users_checkins: pd.DataFrame):
        users_checkins_sorted = users_checkins.sort_values(by=["datetime"])
        
        users_ids = users_checkins_sorted["userid_int"]
        datetimes = pd.to_datetime(users_checkins["datetime"])
        total_users = len(users_checkins["userid_int"].unique())

        self._user_time_frequency = np.zeros((total_users, 48))

        for i,j in zip(users_ids, datetimes):
            if(j.weekday() >= 5):
                self._user_time_frequency[i][j.hour + 24] += 1
            else:
                self._user_time_frequency[i][j.hour] += 1

    def _create_location_coocurrency_matrix(self, users_checkins):
        try:
            users_checkins_sorted = users_checkins.sort_values(by=["datetime"])
            locations = users_checkins_sorted["placeid_int"].tolist()
            number_of_locations = len(users_checkins["placeid_int"].unique())

            self._location_co_ocurrency = sparse.lil_matrix((number_of_locations, number_of_locations)) ##location co occurency represents memory for save memory
            
            for i in range(len(locations)):
                for j in range(1, 6):
                    if((i - j) < 0):
                        break
                    self._location_co_ocurrency[locations[i], locations[i - j]] += 1
                for j in range(1, 6):
                    if(i + j) > len(locations) - 1:
                        break
                    self._location_co_ocurrency[locations[i], locations[j + i]] += 1
            sum_of_dl = np.sum(self._location_co_ocurrency)
            l_occurrency = np.sum(self._location_co_ocurrency, axis=1).reshape(-1, 1)
            c_occurrency = np.sum(self._location_co_ocurrency, axis=0).reshape(1, -1)
            

            for i in range(number_of_locations):
                line = self._location_co_ocurrency[i].toarray()
                ##PMI em subdivisoes da matriz esparsa
                self._location_co_ocurrency[i] = np.maximum(np.log2(np.maximum(line * sum_of_dl, 1)/(l_occurrency[i] * c_occurrency)), 0)

           
            ##Non-sparse implementation
            """ Dl = np.zeros((number_of_locations, number_of_locations))
            for i in range(len(locations)):
                for j in range(1, 6):
                    if((i - j) < 0):
                        break
                    Dl[locations[i], locations[i - j]] += 1
                for j in range(1, 6):
                    if(i + j) > len(locations) - 1:
                        break
                    Dl[locations[i], locations[j + i]] += 1
            sum_of_dl = np.sum(Dl)
            l_occurrency = np.sum(Dl, axis=1).reshape(-1, 1)
            c_occurrency = np.sum(Dl, axis=0).reshape(1, -1)
            teste = np.maximum(np.log2(np.maximum(Dl * sum_of_dl, 1)/(l_occurrency * c_occurrency)), 0)
            
            print((self._location_co_ocurrency[1].toarray() == teste[1]).all())
            print("FIM")         """
        except Exception as e:
            raise e

        

    def _create_location_time_matrix(self, users_checinks):
        
        locations = users_checinks["placeid_int"].tolist()
        datetimes = users_checinks["datetime"].tolist()
        total_locations = len(users_checinks["placeid_int"].unique())
        Dt = np.zeros((total_locations, 48))

        for i in range(len(locations)):
            if(datetimes[i].weekday() >= 5):
                Dt[locations[i]][datetimes[i].hour + 24] += 1
            else: 
                Dt[locations[i]][datetimes[i].hour] += 1

        sum_of_dt = np.sum(Dt)
        l_occurrency = np.sum(Dt, axis=1).reshape(-1, 1)
        c_occurrency = np.sum(Dt, axis=0).reshape(1, -1)
        
        self._location_time = np.maximum(np.log2(np.maximum(Dt * sum_of_dt, 1)/(l_occurrency * c_occurrency)), 0)

                

    def _objective_function(self, weightL):
        
        first_equation = weightL * norm((self._user_location_frequency - np.dot(self.user_activity, self.activity_location.T)))
        second_equation = (1 - weightL) * norm((self._user_time_frequency - np.dot(self.user_activity, self.activity_time.T )))
        activity_modeling_component = first_equation + second_equation

        first_equation = weightL * norm((self._location_co_ocurrency - np.dot(self.target_Location_embedding, self.context_location_embedding.T)))
        second_equation = (1 - weightL) * norm((self._location_time - np.dot(self.target_Location_embedding, self.time_slot_embedding.T )))
        trajectory_embedding_component = first_equation+second_equation

        first_equation = weightL * norm((self.activity_location - np.dot(self.context_location_embedding, self.activity_embedding.T)))
        second_equation = (1 - weightL) * norm((self.activity_time - np.dot(self.time_slot_embedding, self.activity_embedding.T)))
        collaborative_learning_component = first_equation + second_equation

        objective_function =  activity_modeling_component + trajectory_embedding_component + collaborative_learning_component
        objective_function += self._weight * norm(self.user_activity) 
        objective_function += self._weight * norm(self.activity_time)
        objective_function += self._weight * norm(self.activity_embedding)
        objective_function += self._weight * norm(self.activity_location)
        objective_function += self._weight * norm(self.context_location_embedding)
        objective_function += self._weight * norm(self.target_Location_embedding)
        objective_function += self._weight * norm(self.time_slot_embedding)

        return objective_function
    
    def _initialize_parameters(self, checkins, K, M):
        total_locations = len(checkins["placeid_int"].unique())
        total_users = len(checkins["userid_int"].unique())
        time_slot = 48

        self.activity_location = np.random.normal(size = (total_locations, K))
        self.activity_time = np.random.normal(size = (time_slot, K))
        self.user_activity = np.random.normal(size = (total_users, K))
        self.activity_embedding = np.random.normal(size = (K, M))
        self.target_Location_embedding = np.random.normal(size = (total_locations, M))
        self.context_location_embedding = np.random.normal(size = (total_locations, M))
        self.time_slot_embedding = np.random.normal(size = (time_slot, M))
    
    def _optimize_parameters(self, K, M, weightL):
        first_equation = (weightL * (self._user_location_frequency * self.activity_location)) + ((1 - weightL) * np.dot(self._user_time_frequency, self.activity_time)) 
        second_equation = (weightL * np.dot(self.activity_location.T, self.activity_location)) + ((1 - weightL) * np.dot(self.activity_time.T, self.activity_time) + (weightL * np.identity(K)))
        self.user_activity = np.dot(first_equation, inverse(second_equation))

        first_equation = weightL * ((self._user_location_frequency.T * self.user_activity) + np.dot(self.context_location_embedding, self.activity_embedding.T))
        second_equation = (weightL * np.dot(self.user_activity.T, self.user_activity)) + ((self._weight + weightL) * np.identity(K))
        self.activity_location = np.dot(first_equation, inverse(second_equation))

        first_equation = (1 - weightL) * (np.dot(self._user_time_frequency.T, self.user_activity) + np.dot(self.time_slot_embedding, self.activity_embedding.T))
        second_equation = (1- weightL) * (np.dot(self.user_activity.T, self.user_activity) + (1 - self._weight + weightL) * np.identity(K))
        self.activity_time = np.dot(first_equation, inverse(second_equation))

        first_equation = (weightL * np.dot(self.activity_location.T, self.context_location_embedding)) + ((1 - weightL) * np.dot(self.activity_time.T, self.time_slot_embedding))
        second_equation = (weightL * np.dot(self.context_location_embedding.T, self.context_location_embedding)) + ((1 - weightL) * np.dot(self.time_slot_embedding.T, self.time_slot_embedding)) + (self._weight * np.identity(M))
        self.activity_embedding = np.dot(first_equation, inverse(second_equation))

        first_equation = (weightL * (self._location_co_ocurrency * self.context_location_embedding)) + ((1 - weightL) * np.dot(self._location_time, self.time_slot_embedding))
        second_equation = (weightL * np.dot(self.context_location_embedding.T, self.context_location_embedding)) + ((1 - weightL) * np.dot(self.time_slot_embedding.T, self.time_slot_embedding)) + (self._weight * np.identity(M))
        self.target_Location_embedding = np.dot(first_equation, inverse(second_equation))

        first_equation = weightL * (self._location_co_ocurrency.T * self.target_Location_embedding + np.dot(self.activity_location, self.activity_embedding))
        second_equation = (weightL * (np.dot(self.target_Location_embedding.T, self.target_Location_embedding) + np.dot(self.activity_embedding.T, self.activity_embedding))) + (self._weight * np.identity(M))
        self.context_location_embedding = np.dot(first_equation, inverse(second_equation))

        first_equation = (1 - weightL) * (np.dot(self._location_time.T, self.target_Location_embedding) + np.dot(self.activity_time, self.activity_embedding))
        second_equation = ((1 - weightL) * (np.dot(self.target_Location_embedding.T, self.target_Location_embedding) + np.dot(self.activity_embedding.T, self.activity_embedding))) + (self._weight * np.identity(M))
        self.time_slot_embedding = np.dot(first_equation, inverse(second_equation))

        

    def start(self, checkins, weightL = 0.1, K = 10, M = 100):
        checkins["datetime"] = pd.to_datetime(checkins["datetime"])
        checkins["placeid_int"] = checkins["placeid_int"].astype("int")


        self._create_user_location_frequency_matrix(checkins)
        self._create_location_coocurrency_matrix(checkins)
        self._create_user_time_frequency_matrix(checkins)
        self._create_location_time_matrix(checkins)

        self._initialize_parameters(checkins, K, M)
        value = 100000
        print(self.target_Location_embedding.shape)
        for i in range(10):
            self._optimize_parameters(K, M, weightL)
            objective_func = self._objective_function(weightL)
            if((value - objective_func) <= 0.1):
                break
            value = objective_func              

        """self._weight = weight
        
        objective_stable = False
        objective_function_list = []
        while( not objective_stable ):
            self._optimize_parameters(K, M)
            
            objective_function_value = self._objective_function()
            objective_function_list.append(objective_function_value)

            if(len(objective_function_list) == 10):
                objective_function_deviation = np.std(objective_function_list)
                if(objective_function_deviation <= 0.0001):
                    return
                else:
                    objective_function_list = []
         """
    
        
    







    
        
        
        
        






