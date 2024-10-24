#if (a != b) or ((a,b) != (0,len(cities_pos)-1)):
            #    
            #    e_Kprime = norm(cities_pos[(a-1)%len(cities_pos)] -cities_pos[(b)%len(cities_pos)]) + norm(cities_pos[(a)%len(cities_pos)] -cities_pos[(b+1)%len(cities_pos)]) 
            #    e_K = (norm(cities_pos[(a-1)%len(cities_pos)]- cities_pos[(a)%len(cities_pos)]) + norm(cities_pos[(b)%len(cities_pos)]- cities_pos[(b+1)%len(cities_pos)]))
            #    dE = e_Kprime - e_K   
            #    t += dE
            #    if np.exp(-dE/current_temp) > np.random.random():
            #        
            #        cities_pos = change_pos(cities_pos,a,b)
            #        #print("current dE", dE)
            #        #print("current_E + dE", current_E + dE)
            #        
            #        current_E = current_E + dE
            #        all_energies[m_s,n_s] = current_E
            #    else:
            #        all_energies[m_s,n_s] = current_E
            #        #print("current dE", current_E)  
            #else:
            #    all_energies[m_s,n_s] = current_E
            #    #print("current dE", current_E) 
            ##print("current", current_E)