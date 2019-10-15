import numpy
# scipy.special for the sigmoid function expit()
import scipy.special
# neural network class definition
#coding=utf-8
class neuralNetwork:
    
    
    # initialise the neural network
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        # set number of nodes in each input, hidden, output layer
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        
        # link weight matrices, wih and who
        # weights inside the arrays are w_i_j, where link is from node i to node j in the next layer
        # w11 w21
        # w12 w22 etc 
        self.wih = numpy.random.normal(0.0, pow(self.inodes, -0.5), (self.hnodes, self.inodes))
        self.who = numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.onodes, self.hnodes))

        # learning rate
        self.lr = learningrate
        
        # activation function is the sigmoid function
        self.activation_function = lambda x: scipy.special.expit(x)
        
        pass

    
    # train the neural network
    def train(self, inputs_list, targets_list):
        # convert inputs list to 2d array
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T
        
        # calculate signals into hidden layer
        hidden_inputs = numpy.dot(self.wih, inputs)
        # calculate the signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)
        
        # calculate signals into final output layer
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # calculate the signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)
        
        # output layer error is the (target - actual)
        output_errors = targets - final_outputs
        # hidden layer error is the output_errors, split by weights, recombined at hidden nodes
        hidden_errors = numpy.dot(self.who.T, output_errors) 
        
        # update the weights for the links between the hidden and output layers
        self.who += self.lr * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)), numpy.transpose(hidden_outputs))
        
        # update the weights for the links between the input and hidden layers
        self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), numpy.transpose(inputs))
        
        pass

    
    # query the neural network
    def query(self, inputs_list):
        # convert inputs list to 2d array
        inputs = numpy.array(inputs_list, ndmin=2).T
        
        # calculate signals into hidden layer
        hidden_inputs = numpy.dot(self.wih, inputs)
        # calculate the signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)
        
        # calculate signals into final output layer
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # calculate the signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)
        
        return final_outputs


# number of input, hidden and output nodes
input_nodes = 4


#hidden_nodes = 3
hidden_nodes = 3
output_nodes = 7

# learning rate is 0.3
learning_rate = 0.7
#learning_rate = 0.3

# create instance of neural network
n = neuralNetwork(input_nodes,hidden_nodes,output_nodes, learning_rate)

class dataPre:
    def __init__(self,training_data_list, *args, **kwargs):
        
        return super().__init__(*args, **kwargs)

        
    usefulIndex = []
    #gr自然伽玛
    grs = []
    #rd, 深测向的斜率
    slope_rd = []
    #rs, 浅测向的斜率
    slope_rs = []
    targets = []
    depths = []
    grNormalization= 0 

    def slopeOpera(self,up,down,upDepth,downDepth):
        post =(float(down) - float(up))/(float(downDepth) - float(upDepth))
        return post
    def depthOpera(self,upDepth,downDepth):
        post = float(downDepth) - float(upDepth)
        return post
    #取出所有深侧向潜侧向，算出所有数据中的斜率。
    def slope(self,training_data_list):

        #取出训练集数据最大值-1，算出所有点之间的斜率。
        slpoeSize = len(training_data_list) - 1
        for now in range(0,slpoeSize):

            tmps = training_data_list[now].split(',')
            next = training_data_list[now+1].split(',')
            #0是井深，3是潜侧向，2是深侧向，1是自然伽马。
            #当深度或者电阻率的差为0时数据就不应该加入训练集。
            if float(next[3]) - float(tmps[3]) != 0 and float(next[0]) - float(tmps[0]) != 0 and float(next[2]) - float(tmps[2]) != 0 and float(next[0]) - float(tmps[0]) < 100 :
                slopeRs = self.slopeOpera(tmps[3],next[3],tmps[0],next[0])
                self.slope_rs.append(slopeRs)
                depth = self.depthOpera(tmps[0],next[0])
                self.depths.append(depth)
                self.usefulIndex.append(now)
                #print(slopeRs)
            #else:
                #del(training_data_list[now])
                #slpoeSize -= 1
    
            if float(next[2]) - float(tmps[2]) != 0 and float(next[0]) - float(tmps[0]) != 0 and float(next[3]) - float(tmps[3]) != 0 and float(next[0]) - float(tmps[0]) < 100 :    
                slopeRd = self.slopeOpera(tmps[2],next[2],tmps[0],next[0])
                self.slope_rd.append(slopeRd)
                #print(slopeRd)
            #else:
                #del(training_data_list[now])
                #slpoeSize -= 2
    #将所有的gr数据取出，放在grs数组中。
    def grOpera(self):
        for index in self.usefulIndex:
            all_values = training_data_list[index].split(',')
            self.grs.append(float(all_values[1]))
    
    def targetOpera(self):
        for index in self.usefulIndex:
            all_values = training_data_list[index].split(',')
            self.targets.append(float(all_values[4]))

               
            
    #为了实现归一化，需要先计算出最大值与最小值的差。
    def grNormalization(self):
        maxi = max(self.grs)

        Normal = max(self.grs) - min(self.grs)
        for index in range(0,len(self.grs)):
            self.grs[index] = (maxi - self.grs[index]) / Normal
        #for gr in self.grs:
            #gr = (maxi - gr) / Normal
            
        #return Normal
        #print(self.grNormal)
        #print(max(self.grs))
        #print(min(self.grs))
        #pass
    def rdNormalization(self):
        maxi = max(self.slope_rd)

        Normal = max(self.slope_rd) - min(self.slope_rd)
        for index in range(0,len(self.slope_rd)):
            self.slope_rd[index] = (maxi - self.slope_rd[index]) / Normal        
    def rsNormalization(self):
        maxi = max(self.slope_rs)

        Normal = max(self.slope_rs) - min(self.slope_rs)
        for index in range(0,len(self.slope_rs)):
            self.slope_rs[index] = (maxi - self.slope_rs[index]) / Normal    

    def targetNormalization(self):
        self.targets
        pass
        
    def depthNormalization(self):
        maxi = max(self.depths)
        Normal = max(self.depths) - min(self.depths)
        for index in range(0,len(self.depths)):
            self.depths[index] = (maxi - self.depths[index]) / Normal    


#print(n.query([1.0, 0.5, -1.5]))

# load the mnist training data CSV file into a list
training_data_file = open("wellallTrainOri.csv", 'r')
#training_data_file = open("mnist_dataset/mnist_train.csv", 'r')
training_data_list = training_data_file.readlines()
training_data_file.close()

pre = dataPre(training_data_list)
pre.slope(training_data_list)
#print(max(pre.slope_rd))
pre.grOpera()
pre.targetOpera()
pre.grNormalization()
pre.rdNormalization()
pre.rsNormalization()
pre.depthNormalization()





'''
rdCount = len(pre.slope_rd)
rsCount = len(pre.slope_rs)
targetCount = len(pre.targets)
grCount = len(pre.grs)
print(rdCount,rsCount,targetCount,grCount)

def slope(up,down,upDepth,downDepth):
    post =(float(down) - float(up))/(float(downDepth) - float(upDepth))
    return post
    print(pre.targets[0])
#for x in slope_rs:
    #print(x)
for pre in training_data_list:
    tmps = pre.split(',')
    grs.append(float(tmps[1]))
#print(grs)
#print(max(grs))
#print(min(grs))

#data pretreatment
for cou in range(0,len(training_data_list)):
    for  pres in training_data_list[cou]:
        if pres == '0':
            
            del(training_data_list[cou])
for eve in training_data_list:
    print(eve)
 

#print(training_data_list[0])
# train the neural network
# epochs is the number of times the training data set is used for training


    # go through all records in the training data set
    for record in training_data_list:
        # split the record by the ',' commas
        all_values = record.split(',')
        # scale and shift the inputs
        inputs = (numpy.asfarray(all_values[1:3]) / 2000.0 * 0.99) + 0.01
        # create the target output values (all 0.01, except the desired label which is 0.99)
        targets = numpy.zeros(output_nodes) + 0.01
        # all_values[0] is the target label for this record
        targets[int(all_values[4])] = 0.99
        #print(inputs)
        #print(targets)
        n.train(inputs, targets)
        pass
    pass
'''

epochs = 8
ext = len(pre.slope_rs) -1
for e in range(epochs):
    for i in range(0,ext):
        #inputs = pre.slope_rd[i] + pre.slope_rs[i] + pre.grs[i]
        inputs = []
        inputs.append(pre.depths[i])
        inputs.append(pre.slope_rd[i])
        inputs.append(pre.slope_rs[i])
        inputs.append(pre.grs[i])
        targets = numpy.zeros(output_nodes) + 0.01
        targets[int(pre.targets[i])] = 0.99
        #print(inputs,targets)
        n.train(inputs,targets)
        pass
    pass

# load the mnist test data CSV file into a list
test_data_file = open("well89.csv", 'r')
test_data_list = test_data_file.readlines()
test_data_file.close()
# test the neural network

tes = dataPre(test_data_list)
tes.slope(test_data_list)
#print(max(pre.slope_rd))
tes.grOpera()
tes.targetOpera()
tes.grNormalization()
tes.rdNormalization()
tes.rsNormalization()
tes.depthNormalization()


# scorecard for how well the network performs, initially empty
scorecard = []
extTes = len(tes.slope_rs)
for i in range(0,extTes):
    correct_label = int(tes.targets[i])
    inputs = []
    inputs.append(tes.depths[i])
    inputs.append(tes.slope_rd[i])
    inputs.append(tes.slope_rs[i])
    inputs.append(tes.grs[i])
   #print(inputs,correct_label)
    outputs = n.query(inputs)
    label = numpy.argmax(outputs)
    if (label == correct_label):
        # network's answer matches correct answer, add 1 to scorecard
        scorecard.append(1)
    else:
        # network's answer doesn't match correct answer, add 0 to scorecard
        scorecard.append(0)
        pass
    
    pass



scorecard_array = numpy.asarray(scorecard)
#print ("performance = ", scorecard_array.sum() / scorecard_array.size)
print ("正确率 = ", scorecard_array.sum() / scorecard_array.size * 100)
