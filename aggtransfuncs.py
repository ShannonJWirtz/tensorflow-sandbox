from tfsource import *
from datasource import *
from stocgenfuncs1 import *

class Tftrans(object):

	def __init__(self, data, input_scaling_vars, input_encoding_vars, output_vars, partition_name = 'partition'):
		self.input_scaling_vars = [input_scaling_vars] if isinstance(input_scaling_vars, str) else input_scaling_vars
		self.input_encoding_vars = [input_encoding_vars] if isinstance(input_encoding_vars, str) else input_encoding_vars
		self.input_vars = self.input_scaling_vars + self.input_encoding_vars
		self.output_vars = [output_vars] if isinstance(output_vars, str) else output_vars
		self.input_encoding_drop_values = data[self.input_encoding_vars].iloc[0]
		self.input_encoding_drop_columns = [str(a)+'_'+str(b) for a,b in zip(self.input_encoding_vars, self.input_encoding_drop_values)]
		self.sscaler = preprocessing.StandardScaler()
		self.scaling_vars_fit = self.sscaler.fit(data[self.input_scaling_vars])
		input_data_scaled = data[self.input_vars]
		scaled_vars_transformed = self.scaling_vars_fit.transform(input_data_scaled[self.input_scaling_vars])
		input_data_scaled[self.input_scaling_vars] = scaled_vars_transformed

		input_data_scaled_encoded_preselect = pd.get_dummies(input_data_scaled, columns = self.input_encoding_vars)

		input_data_scaled_encoded = input_data_scaled_encoded_preselect.drop(self.input_encoding_drop_columns, axis = 1)

		self.inputs_array = np.array(input_data_scaled_encoded_preselect)
		self.outputs_array = np.array(data[self.output_vars])




class Agg(object):

	def __init__(self, sample, n, trainprop, input_scaling_vars, input_encoding_vars, output_vars):
		self.data = []
		self.sample = sample
		self.trainlen =  math.floor(n*trainprop)
		self.n = n
		for i in range(n):
			observation = self.sample()
			observation['partition'] = 'train' if i < self.trainlen else 'test'
			self.data.append(observation)
		self.data = pd.DataFrame(self.data)

		self.tft = Tftrans(self.data, input_scaling_vars, input_encoding_vars, output_vars)
		self.train_inputs_array = self.tft.inputs_array[range(self.trainlen)]
		self.train_outputs_array = self.tft.outputs_array[range(self.trainlen)]
		self.test_inputs_array = self.tft.inputs_array[self.trainlen:n]
		self.test_outputs_array = self.tft.outputs_array[self.trainlen:n]



