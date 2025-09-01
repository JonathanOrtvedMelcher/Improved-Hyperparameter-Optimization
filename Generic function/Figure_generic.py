#This program generates the figure for the simple function, as well as results used for table 1, and some other numerical
#values used in the text

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import random
import math
import pickle
import math

plt.style.use('classic')

matplotlib.rcParams['font.family']='sans-serif'
matplotlib.rcParams['font.sans-serif']=['DejaVu Serif']
matplotlib.rcParams['mathtext.fontset'] = 'dejavuserif'

################accessing the results

#index 1 of lists with data corresponds to the grid pattern
#index 2 of lists with data corresponds to the random pattern
#index 3 of lists with data corresponds to the gridrandom pattern


def do_plot(d, ax, position, incl=0, k=1):

    if d==2:
       mak=8
    if d==3 or d==4:
       mak=5
    if d>4:
       mak=3

    upper_limit = mak+4

    list_good = [1, 2, mak+3]

    list_good2 = []
    if d==2:
        for i in range(1, upper_limit):
            if i!=4 and i!=6 and i!=7 and i!=8:
                list_good2.append(i)
    else:
        list_good2 = []
        for i in range(1, upper_limit):
            if not ((d==3 and i==4) or (d==4 and i==4)):
                list_good2.append(i)

    file_scores = "Scores_d_" + str(d) + ".txt"
    file_numbers = "Numbers_d_" + str(d) + ".txt"


    everything_scores = pickle.load(open(file_scores, 'rb'))
    everything_numbers = pickle.load(open(file_numbers, 'rb'))

    def stealth_hyper(i):
        if d==2:
            if i==0:
                return "0.49"
            if i==1:
                return "0.38"
            if i==2:
                return "0.25"
            if i==3:
                return "0.13"
        else:
            if i==0:
                return "0.4"
            if i==1:
                return "0.3"

    #number of stealths
    if d==2:
        maks_hyper = 4
    else:
        maks_hyper = 2

    hyper_scores = [[] for i in range(maks_hyper)]
    hyper_numbers = [[] for i in range(maks_hyper)]

    for i in range(maks_hyper):
        file_scores_hyper = "Hyperuniform_scores_" + stealth_hyper(i) + "_d_" + str(d) + ".txt"
        file_numbers_hyper = "Hyperuniform_numbers_" + stealth_hyper(i) + "_d_" + str(d) + ".txt"

        hyper_scores[i] = pickle.load(open(file_scores_hyper, 'rb'))
        hyper_numbers[i] = pickle.load(open(file_numbers_hyper, 'rb'))

    file_scores_random = "Random_scores_d_" + str(d) + ".txt"
    file_numbers_random = "Random_numbers_d_" + str(d) + ".txt"

    random_scores = pickle.load(open(file_scores_random, 'rb'))
    random_numbers = pickle.load(open(file_numbers_random, 'rb'))
    
    if d==3:
        file_scores_quasi = r"Quasiperiodic_scores_d_3_star_5.txt"
        file_numbers_quasi = r"Quasiperiodic_numbers_d_3_star_5.txt"

    else:
        file_scores_quasi = r"Quasiperiodic_scores_d_4_star_6.txt"
        file_numbers_quasi = r"Quasiperiodic_numbers_d_4_star_6.txt"
   
    quasi_scores = pickle.load(open(file_scores_quasi, 'rb'))
    quasi_numbers = pickle.load(open(file_numbers_quasi, 'rb'))


    numbers = everything_numbers[-1]

    numbers_h = [[] for i in range(maks_hyper)]
    numbers_r = []
    numbers_q = []
    
    errors_numbers_h = [[] for i in range(maks_hyper)]


    N = len(numbers[1])+1

    d = int(math.log2(numbers[1][-1]))

    scores = [[0] for i in range(upper_limit)]
    scores_h = [[] for i in range(maks_hyper)]

    results = [[] for i in range(upper_limit)] #overall results for a given number of points

    results_h = [[] for i in range(maks_hyper)]


    listquasiall = [[] for i in range(mak+1)]
    lengths = [0 for i in range(mak+1)]


    rev = [(N-i) for i in range(N-1)]

    y_err = [[] for i in range(upper_limit)]

    y_err_h = [[] for i in range(maks_hyper)]

    x_err = [[] for i in range(upper_limit)]

    wins = [[0 for i in range(upper_limit)] for j in range(len(rev)+2)]
    wins_hyper = [[0 for i in range(maks_hyper)] for j in range(len(rev)+2)]
    wins2 = [[0 for i in range(upper_limit)] for j in range(len(rev)+2)]


    new_min = np.array([-k/2 for i in range(d)])
    new_max = np.array([k/2 for i in range(d)])

    old_min = np.array([0 for i in range(d)])
    old_max = np.array([k for i in range(d)])


    counter = 0

    def tag(i):
      if i==1:
          return "Grid"
      if i==2:
          return "Random"
      if i==3:
          return "Gridrandom"
      if i>=4 and i<=mak:
          return "quasi_" + str(i)
      if i==mak+1:
          return "RSA"
      if i==mak+2:
          return "Latin"
      if i==mak+3:
          return "Hyper"

    def tag_long(i):
      if i==1:
          return "Grid"
      if i==2:
          return "Random"
      if i==3:
          return "Gridrandom"
      if d==2 and i==5:
          return "Penrose"
      if i>=4 and i<=mak:
          return "Quasiperiodic"
      if i==mak+1:
          return "RSA"
      if i==mak+2:
          return "Latin Hypercube Sampling"
      if i==mak+3:
          return "Hyperuniform"


    for n in rev:

            for i in range(maks_hyper):
                numbers_h[i].append(sum(hyper_numbers[i][counter][0])/len(hyper_numbers[i][counter][0]))
                errors_numbers_h[i].append(np.std(hyper_numbers[i][counter][0])/np.sqrt(len(hyper_numbers[i][counter][0])))
                scores_h[i] = hyper_scores[i][counter]
            
            scores = everything_scores[counter]

            if d==2:
                scores[-1] = scores_h[0][0]
            else:
                scores.append(scores_h[0][0])

            scores_r = random_scores[counter][0]
            scores[2] = scores_r
            numbers_r.append(sum(random_numbers[counter][0])/len(random_numbers[counter][0]))
            if d==3 or d==4:
                numbers_q.append(sum(quasi_numbers[counter][0])/len(quasi_numbers[counter][0]))

            if d==3 or d==4:
                scores = scores[:4] + [[], quasi_scores[counter][0]] + scores[4:]

            for j in range(len(scores[1])):
                  listhelp = []
                  listhelp2 = []
                  listhelp_hyper = []
                  for i in range(1, len(scores)):
                    if i in list_good:
                      try:
                          listhelp.append(scores[i][j])
                      except IndexError:
                          listhelp.append(scores[1][j] + 1)
                    else:
                        listhelp.append(scores[1][j] + 1)

                    if i in list_good2:
                         try:
                            listhelp2.append(scores[i][j])
                         except IndexError:
                            listhelp2.append(scores[1][j] + 1)
                    else:
                        listhelp2.append(scores[1][j] + 1)


                  for i in range(maks_hyper):
                        listhelp_hyper.append(scores_h[i][0][j])
                        
                  wins[n][listhelp.index(min(listhelp))+1] += 1

                  
                  wins2[n][listhelp2.index(min(listhelp2))+1] += 1
                  
                  wins_hyper[n][listhelp_hyper.index(min(listhelp_hyper))] += 1
              
            for i in range(1, upper_limit):
              if len(scores[i])>=1:
                results[i].append(sum(scores[i])/len(scores[i]))

            for i in range(maks_hyper):    
                results_h[i].append(sum(scores_h[i][0])/len(scores_h[i][0]))
                    
            y_err[1].append(np.std(scores[1])/np.sqrt(len(scores[1])))
            y_err[2].append(np.std(scores[2])/np.sqrt(len(scores[2])))


            if len(scores[3])>=1:
              y_err[3].append(np.std(scores[3])/np.sqrt(len(scores[3])))

            for i in range(4, mak+1):
              if len(scores[i])>=1:
                y_err[i].append(np.std(scores[i])/np.sqrt(len(scores[i])))

            i = mak+1  
            y_err[i].append(np.std(scores[i])/np.sqrt(len(scores[i])))


            i = mak+2
            y_err[i].append(np.std(scores[i])/np.sqrt(len(scores[i])))
            
            i = mak+3
            y_err[i].append(np.std(scores[i])/np.sqrt(len(scores[i])))

                
            for i in range(maks_hyper):
                y_err_h[i].append(np.std(scores_h[i][0])/np.sqrt(len(scores_h[i][0])))

            counter += 1

    if d==2:
        numbers[-1] = numbers_h[0]
    else:
        numbers.append(numbers_h[0])
    numbers[2] = numbers_r

    if d==3 or d==4:
        numbers = numbers[:4] + [[], numbers_q] + numbers[4:]

    impr_grid = [0 for i in range(upper_limit)]
    impr_rand = [0 for i in range(upper_limit)]

    impr_h_grid = [0]
    impr_h_rand = [0]

    for i in range(1, upper_limit):
        if len(results[i])>0:
            impr_grid[i] = 100*sum([(results[i][j] - results[1][j])/results[1][j] for j in range(len(results[i]))])/len(results[i])
            impr_rand[i] = 100*sum([(results[i][j] - results[2][j])/results[2][j] for j in range(len(results[i]))])/len(results[i])


    impr_h_grid[0] = 100*sum([(results_h[j] - results[1][j])/results[1][j] for j in range(len(results_h))])/len(results_h)
    impr_h_rand[0] = 100*sum([(results_h[j] - results[2][j])/results[2][j] for j in range(len(results_h))])/len(results_h)

    print("---------------------------------------------------------")
    print(str(d) + " dimensions - results for the simple function used in the manuscript \n")

    print("Improvement of hyperuniform over grid search in %: ", -impr_grid[mak+3])
    print("Improvement of hyperuniform over random search in %: ", -impr_rand[mak+3])

    i = 1
    wins_grid[0] += sum(wins[j][i] for j in range(len(wins)))

    i = 2
    wins_random[0] += sum(wins[j][i] for j in range(len(wins)))

    i = mak+3
    wins_hyperuniform[0] += sum(wins[j][i] for j in range(len(wins)))

    for i in range(2, upper_limit):
        for j in range(len(results[i])):
            results[i][j] = results[i][j]/results[1][j]
            y_err[i][j] = results[i][j] * np.sqrt((y_err[i][j]/(results[i][j]*results[1][j]))**2 + (y_err[1][j]/results[1][j])**2)

    for i in range(maks_hyper):
        for j in range(len(results_h[i])):
            results_h[i][j] = results_h[i][j]/results[1][j]
            y_err_h[i][j] = results_h[i][j] * np.sqrt((y_err_h[i][j]/(results_h[i][j]*results[1][j]))**2 + (y_err_h[i][j]/results[1][j])**2)

    i=1
    for j in range(len(results[i])):
        results[i][j] = results[i][j]/results[1][j]
        y_err[i][j] = 0

    ###printing results for the tables


    print("---------------------------------------------------------")

    if d==2:
        number_maximal = 15
    else:
        number_maximal = max(rev)



    print(str(d) + " dimensions - results for the simple function used in Tables I and IV \n")


    avg_all = [0]
    for i in range(1, upper_limit):
        avg_all.append(sum(results[i][-(number_maximal-1):])/(number_maximal-1))
    wins_comb = [0 for i in range(upper_limit)]
    for i in range(upper_limit):
        wins_comb[i] = sum(wins2[j][i] for j in range(2, number_maximal+1))
    frac_wins = [wins_comb[i]/sum(wins_comb) for i in range(upper_limit)]

    print("Average loss for each strategy")
    for i in list_good2:
        print(tag_long(i), avg_all[i])
    print("")

    print("Fraction of times each strategy was best")
    for i in list_good2:
        print(tag_long(i), frac_wins[i])

    print('\n')


    avg_all_hyper = []
    for i in range(maks_hyper):
        avg_all_hyper.append(sum(results_h[i][-(number_maximal-1):])/(number_maximal-1))
    wins_comb_hyper = [0 for i in range(maks_hyper)]
    for i in range(maks_hyper):
        wins_comb_hyper[i] = sum(wins_hyper[j][i] for j in range(2, number_maximal+1))
    frac_wins_hyper = [wins_comb_hyper[i]/sum(wins_comb_hyper) for i in range(maks_hyper)]

    print("Average loss for each strategy (hyperuniform)")
    for i in range(maks_hyper):
        print(stealth_hyper(i), "- hyperuniform", avg_all_hyper[i])
    print("")

    print("Fraction of times each strategy was best (hyperuniform)")
    for i in range(maks_hyper):
        print(stealth_hyper(i), "- hyperuniform", frac_wins_hyper[i])

    #print(wins_comb_hyper)


    tilings = ['grid', 'random', 'gridrandom']
    for i in range(4, mak+1):
               tilings.append("quasi_" + str(i))
    tilings += ['RSA', 'Latin', 'Hyperuniform']


    colors = ["black", "red", "magenta", "olive", "yellow", "olivedrab", \
             "darkolivegreen", "forestgreen", "turquoise", "chocolate", "teal", \
             "deepskyblue", "slategray", "navy", "blueviolet", "violet", \
             "purple", "deeppink", "crimson"]

    size = 2


    def which_color(i):
        if i==1:
            return "green"
        if i==2:
            return "blue"
        if i==3:
            return "black"
        if i==mak+1:
            return "orange"
        if i==mak+3:
            return "red"
        if i!=1 and i!=2 and i!=3 and i!=mak+1 and i!=mak+3:
            return colors[(3*i)%len(colors)]

    def which_marker(i):
        if i==1:
            return 'o'
        if i==2:
            return '^'
        if i==mak+3:
            return 's'
        else:
            return 'o'

    list_good = [1, 2, mak+3]

    for i in list_good:
      if i==mak+3:
          x_err = errors_numbers_h[0]
      else:
          x_err = [0 for i in range(len(numbers[i]))]
      ax.errorbar(numbers[i], results[i], yerr=y_err[i], xerr=x_err, color=which_color(i), elinewidth=3, capthick=3)
      ax.plot(numbers[i], results[i], marker=which_marker(i), markersize=15, linestyle='-', linewidth=3, color = which_color(i), label=tag_long(i))
      ax.fill_between(numbers[i], [results[i][j]-y_err[i][j] for j in range(len(results[i]))], \
                    [results[i][j]+y_err[i][j] for j in range(len(results[i]))], color = which_color(i), alpha = 0.1)

      
    
    colors_hyper = ['black', 'cyan', 'purple', 'lightpink']

    listgood_hyper = [i for i in range(maks_hyper)]
    listgood_hyper = []

    #Potting results for the hyperuniform patterns
    for i in listgood_hyper:
        ax.errorbar(numbers_h[i], results_h[i], yerr=y_err_h[i], xerr=errors_numbers_h[i], color=colors_hyper[i], elinewidth=3, capthick=3)
        ax.plot(numbers_h[i], results_h[i], 'o-', color = colors_hyper[i], label="Hyper " + stealth_hyper(i), linewidth=3)
        ax.fill_between(numbers_h[i], [results_h[i][j]-y_err_h[i][j] for j in range(len(results_h[i]))], \
                      [results_h[i][j]+y_err_h[i][j] for j in range(len(results_h[i]))], color=colors_hyper[i], alpha = 0.1)



    s = str(d) + "D"
    plt.xlim((min(numbers[mak+1][-1], 2**d) * 0.9, N**d * 1.1))
    if d==3:
        ax.set_ylim((0.45, 1.1))
    if d==2:
        ax.set_ylim((0.42, 1.65))
    if d==4:
        ax.set_ylim((0.45, 1.05))
    if d==5:
        ax.set_ylim((0.43, 1.05))
    ax.set_title(s, fontsize=30, y=0.84, x=0.1)
    ax.set_xscale('log')


    if d==2:
      maks_power = int(np.log10(400*1.1))
    else:
      maks_power = int(np.log10(5**d*1.1))

    values_upper = []
    ticks_upper = []
    
    for i in range(1, maks_power+1):
      if 10**i >= min(numbers[mak+1][-1], 2**d) * 0.9 and 10**i < N**d * 1.1:
        values_upper.append(10**i)
        ticks_upper.append('$10^{}$'.format(i))

    ax.set_xlim((min(numbers[mak+1][-1], 2**d) * 0.9, N**d * 1.1))

    ax.set_xticks(values_upper, ticks_upper, fontsize=25)
    ax.tick_params(labelsize=25)

    if position=='up':
      ax.tick_params('x', bottom = True, top = True, labelbottom = False, labeltop = True, labelsize=25)

    ax.yaxis.set_label_coords(1.05, 0.5)

    ax.set_axisbelow(True)
    ax.set_facecolor('whitesmoke')


    
wins_grid = [0]
wins_random = [0]
wins_hyperuniform = [0]



k=1

do_include = 0 #the old hyperuniform results will not be included


fig, axs = plt.subplots(2, 2, figsize = (7*3, 5*3))

width_grid = 0.2

do_plot(2, axs[0, 0], 'up')
do_plot(3, axs[0, 1], 'up')
do_plot(4, axs[1, 0], 'down')
do_plot(5, axs[1, 1], 'down')

print("---------------------------------------------------------")
print("Results about the number of times each strategy was best for the simple function, used in the manuscript \n")

print("Grid", wins_grid[0])
print("Random", wins_random[0])
print("Hyperuniform", wins_hyperuniform[0])


axs[0, 0].tick_params('y', left = True, right = True, labelleft = True, labelright = False, labelsize=25)
axs[0, 1].tick_params('y', left = True, right = True, labelleft = False, labelright = True, labelsize=25)
axs[1, 0].tick_params('y', left = True, right = True, labelleft = True, labelright = False, labelsize=25)
axs[1, 1].tick_params('y', left = True, right = True, labelleft = False, labelright = True, labelsize=25)


l = axs[1, 1].legend(fontsize=25, markerscale=1.2, loc = 'lower right')
for line in l.get_lines():
    line.set_linewidth(3)


fig.suptitle("Comparison for the simple function in 2-5 dimensions", fontsize=35)
fig.text(-0.02, 0.5, "Loss relative to grid search", va='center', rotation='vertical', fontsize=30)
fig.text(0.5, -0.02, "Number of sampled points", ha='center', rotation='horizontal', fontsize=30)

plt.tight_layout()
plt.savefig("Comparison_simple.jpg", bbox_inches='tight', dpi=600)
plt.show()






















 


