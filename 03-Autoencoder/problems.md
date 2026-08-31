1) -- if you faced a problem with tensorflow and Cuda try this:

    ```python
    !pip install --upgrade pip
    !pip install tensorflow[and-cuda]
    ```

2) -- if you faced a problem with plotting the model try:
    ```python
    !pip install graphviz
    ```
    then if you are using arch linux try:
    ```terminal
    sudo pacman -S graphviz
    ```
    or if you are using Upuntu try:
    ```terminal
    sudo apt-get update
    sudo apt-get install graphviz
    ```
    ----> then it will work -_-