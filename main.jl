# Library of functions used in this script:
include("utils/utils.jl")

using CSV
using DataFrames
using Plots
using Printf

# Load the dataset
document("Loading dataset")
data = CSV.read("datasets/air_quality_health_impact_data.csv", DataFrame)

# Check the general distribution of data in the dataset
document("Dataset characteristics:")
println(describe(data))

# We check how many classes have of each kind
counts = combine(groupby(data, :HealthImpactClass), nrow => :Count)
document("Elements per class:")
println(counts)

# and plot the distribution
cnt = counts.Count
labels = ["Class $i" for i in 0:4]
tot = sum(cnt)
datapct = [@sprintf("%.1f%%", x/tot*100) for x in cnt]

θ = (cumsum(cnt) - cnt/2) .* 360/sum(cnt)
scθ = sincosd.(θ)

p = pie(labels, cnt, title="Output class distribution")
for (s, sci) in zip(datapct, scθ)
    annotate!(1.1*sci[2], 1.1*sci[1], text(s, 8, :black))
end
plot(p, dpi=1000)
savefig("images/class_dist.png")

# Segmentation of data into inputs/outputs
document("Segmentation of data into inputs/outputs")
input_data = Matrix(data[!, 1:13]);
output_data = Int.(data[!, 15]);

@assert input_data isa Matrix
@assert output_data isa Vector{Int64}

# Spliting data into train and test using holdOut
document("Reserving 20% of samples to test")
(tr_idx, test_idx) = holdOutStratified(output_data, 0.2)

train_input = input_data[tr_idx,:]
train_output = output_data[tr_idx]
test_input = input_data[test_idx,:]
test_output = output_data[test_idx]

train_output = collect(train_output)
test_output = collect(test_output)

println("Train Input Size: ", size(train_input))
println("Train Output Size: ", size(train_output), " Categories:", sort(unique(train_output)))
println("Test Input Size: ", size(test_input))
println("Test Output Size: ", size(test_output), " Categories:", sort(unique(test_output)))

# Check if have representation of each classes in train and test
column = train_output[:, 1]
countclasses = Dict(c => count(==(c), column) for c in 0:4)
document("Elements per class in train set:")
println(countclasses)

column = test_output[:, 1]
countclasses = Dict(c => count(==(c), column) for c in 0:4)
document("Elements per class in test set:")
println(countclasses)

# Preparation of empty DataFrame for storing results
metric_names = [:approach, :accuracy, :error_rate, :recall, :specificity, :ppv, :npv, :f_score]
metric_types = [String[], Float64[], Float64[], Float64[], Float64[], Float64[], Float64[], Float64[]]
metrics = DataFrame(metric_types, metric_names)


