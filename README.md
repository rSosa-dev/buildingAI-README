# BuildingAI - My AI idea

This is the final project for the Building AI course.

## My idea

My AI idea will be an image classifier. It can be done with TensorFlow, a python library used for machine learning.
The idea is to simulate a shop and the program will classify the photos we pass by their shape.

## Background

This idea can solve the bad classification of the products of a shop, sometimes the people can make mistakes, and I think that having a classifier would solve some of them.
I think this project can be a good way to start practicing with some of the concept I learnt in this course.

The idea is to compare the pixels of the images to give the AI an idea of what kind of items appears on them.

## How is it used?

This program will be used, of course, when new stock arrives to the shop. The employee will only have to make a photo to every item and put them in the program, and them will be classified automatically.

One example of a function that will be in the program is the next one, here we have an image resizer. To improve the pixel detection, all the images will be shrinked to 100x100px:
```
def graficar_imagen(i, arr_predicciones, etiquetas_reales, imagenes):
  arr_predicciones, etiqueta_real, img = arr_predicciones[i], etiquetas_reales[i], imagenes[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  
  plt.imshow(img[...,0], cmap=plt.cm.binary)

  etiqueta_prediccion = np.argmax(arr_predicciones)
  if etiqueta_prediccion == etiqueta_real:
    color = 'blue'
  else:
    color = 'red'
  
  plt.xlabel("{} {:2.0f}% ({})".format(nombres_clases[etiqueta_prediccion],
                                100*np.max(arr_predicciones),
                                nombres_clases[etiqueta_real]),
                                color=color)
```


## Data sources and AI methods

The program will work with Neural Networks, which will be programmed, as I said, using the open source python library [TensorFlow](https://www.tensorflow.org).


## Challenges

The photos have to be taken on a simple background, the main object must be easily identified, it is a simple classifier.

## What next?

My AI knowledge cannot think any other new implementations to the program. 
