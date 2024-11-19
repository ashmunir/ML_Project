import Pkg;
Pkg.add("Flux")

using Random
using Random:seed!
using Statistics
using Flux
using ScikitLearn
using CSV, DataFrames

@sk_import svm: SVC
@sk_import tree: DecisionTreeClassifier
@sk_import neighbors: KNeighborsClassifier

function oneHotEncoding(feature::AbstractArray{<:Any,1}, classes::AbstractArray{<:Any,1})
    # First we are going to set a line as defensive to check values
    @assert(all([in(value, classes) for value in feature]))
    
    # Second defensive statement, check the number of classes
    numClasses = length(classes)
    @assert(numClasses>1)
    
    if (numClasses==2)
        # Case with only two classes
        oneHot = reshape(feature.==classes[1], :, 1)
    else
        #Case with more than two clases
        oneHot =  BitArray{2}(undef, length(feature), numClasses)
        for numClass = 1:numClasses
            oneHot[:,numClass] .= (feature.==classes[numClass])
        end
    end
    return oneHot
end

# Function to transorm in one-hot-encoding based on the values of the array passed
oneHotEncoding(feature::AbstractArray{<:Any,1}) = oneHotEncoding(feature, unique(feature))
