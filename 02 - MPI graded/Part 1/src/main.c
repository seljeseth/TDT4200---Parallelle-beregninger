#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <getopt.h>
#include <stdlib.h>
#include <time.h>
#include <image_utils.h>
#include <argument_utils.h>
#include <mpi.h>

// NOTE TO STUDENT:
// The kernels are defined under argument_utils.h
// Take a look at this file to get a feel for how the kernels look.

// Apply convolutional kernel on image data
void applyKernel(pixel **out, pixel **in, unsigned int width, unsigned int height, int *kernel, unsigned int kernelDim, float kernelFactor)
{
    unsigned int const kernelCenter = (kernelDim / 2);
    for (unsigned int imageY = 0; imageY < height; imageY++)
    {
        for (unsigned int imageX = 0; imageX < width; imageX++)
        {
            unsigned int ar = 0, ag = 0, ab = 0;
            for (unsigned int kernelY = 0; kernelY < kernelDim; kernelY++)
            {
                int nky = kernelDim - 1 - kernelY;
                for (unsigned int kernelX = 0; kernelX < kernelDim; kernelX++)
                {
                    int nkx = kernelDim - 1 - kernelX;

                    int yy = imageY + (kernelY - kernelCenter);
                    int xx = imageX + (kernelX - kernelCenter);
                    if (xx >= 0 && xx < (int)width && yy >= 0 && yy < (int)height)
                    {
                        ar += in[yy][xx].r * kernel[nky * kernelDim + nkx];
                        ag += in[yy][xx].g * kernel[nky * kernelDim + nkx];
                        ab += in[yy][xx].b * kernel[nky * kernelDim + nkx];
                    }
                }
            }
            if (ar || ag || ab)
            {
                ar *= kernelFactor;
                ag *= kernelFactor;
                ab *= kernelFactor;
                out[imageY][imageX].r = (ar > 255) ? 255 : ar;
                out[imageY][imageX].g = (ag > 255) ? 255 : ag;
                out[imageY][imageX].b = (ab > 255) ? 255 : ab;
                out[imageY][imageX].a = 255;
            }
            else
            {
                out[imageY][imageX].r = 0;
                out[imageY][imageX].g = 0;
                out[imageY][imageX].b = 0;
                out[imageY][imageX].a = 255;
            }
        }
    }
}

int main(int argc, char **argv)
{

    MPI_Init(&argc, &argv);

    int world_sz;
    int world_rank;

    MPI_Comm_size(MPI_COMM_WORLD, &world_sz);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    OPTIONS my_options;
    OPTIONS *options = &my_options;

    if (world_rank == 0)
    {
        options = parse_args(argc, argv);

        if (options == NULL)
        {
            fprintf(stderr, "Options == NULL\n");
            exit(1);
        }
    }

    MPI_Bcast(options, sizeof(OPTIONS), MPI_BYTE, 0, MPI_COMM_WORLD);

    if (world_rank > 0)
    {
        options->input = NULL;
        options->output = NULL;
    }

    image_t dummy;
    dummy.rawdata = NULL;
    dummy.data = NULL;

    image_t *image = &dummy;
    image_t *my_image;

    if (world_rank == 0)
    {
        image = loadImage(options->input);
        if (image == NULL)
        {
            fprintf(stderr, "Could not load bmp image '%s'!\n", options->input);
            freeImage(image);
            abort();
        }
    }

    if (world_rank == 0)
    {
        printf("Apply kernel '%s' on image with %u x %u pixels for %u iterations\n",
               kernelNames[options->kernelIndex],
               image->width,
               image->height,
               options->iterations);
    }

    // Broadcast image information
    MPI_Bcast(image,           // Send Buffer
              sizeof(image_t), // Send Count
              MPI_BYTE,        // Send Type
              0,               // Root
              MPI_COMM_WORLD); // Communicator

    //////////////////////////////////////////////////////////
    // Calculate how much of the image to send to each rank //
    //////////////////////////////////////////////////////////
    int rows_to_receive[world_sz];
    int bytes_to_transfer[world_sz];
    int displacements[world_sz];
    displacements[0] = 0;

    int rows_per_rank = image->height / world_sz;
    int remainder_rows = image->height % world_sz;

    for (int i = 0; i < world_sz; i++)
    {
        int rows_this_rank = rows_per_rank;

        if (i < remainder_rows)
        {
            rows_this_rank++;
        }

        int bytes_this_rank = rows_this_rank * image->width * sizeof(pixel);

        rows_to_receive[i] = rows_this_rank;
        bytes_to_transfer[i] = bytes_this_rank;

        if (i > 0)
        {
            displacements[i] = displacements[i - 1] + bytes_to_transfer[i - 1];
        }
    }

    int num_border_rows = (kernelDims[options->kernelIndex] - 1) / 2;
    int my_image_height = rows_to_receive[world_rank];
    int last_rank = world_sz - 1;

    // When we are at the edge of the image we only need space
    // for the number of border rows in one direction either north or south
    if (world_rank == 0 || world_rank == last_rank)
    {
        my_image = newImage(image->width, my_image_height + num_border_rows);
    }
    // If we are dealing with processes with ranks between 0 and the last rank
    // we need to add space for nr_border_rows both in north and south direction
    else
    {
        my_image = newImage(image->width, my_image_height + (2 * num_border_rows));
    }

    // Ternary operator
    // Every rank other than 0 are not senders and thus
    // do not need to actually have anything in the send buffer. These
    // get their send buffer pointer set to NULL.
    pixel *image_send_buffer = world_rank == 0 ? image->rawdata : NULL;

    // All ranks except the rank 0 will have to skip filling the first row they have allocated
    // If the process has rank 0 we can just write directly to the start.
    // This pointer marks where the actual data starts
    // since we have filled our image buffer with empty rows
    pixel *start_of_image; // Renamed my_image_sliced to start_of_image
    if (world_rank == 0)
    {
        start_of_image = my_image->rawdata;
    }
    else
    {
        start_of_image = my_image->rawdata + (num_border_rows * image->width);
    }

    MPI_Scatterv(image_send_buffer,             // Send Buffer
                 bytes_to_transfer,             // Send Counts
                 displacements,                 // Displacements
                 MPI_BYTE,                      // Send Type
                 start_of_image,                // Recv Buffer
                 bytes_to_transfer[world_rank], // Recv Count
                 MPI_BYTE,                      // Recv Type
                 0,                             // Root
                 MPI_COMM_WORLD);               // Communicator

    // Starting timer
    double starttime = MPI_Wtime();

    // Here we do the actual computation!
    // image->data is a 2-dimensional array of pixel which is accessed row
    // first ([y][x]) each pixel is a struct of 4 unsigned char for the red,
    // blue and green colour channel
    image_t *processImage = newImage(image->width, my_image->height);

    int nr_of_border_pixels = num_border_rows * my_image->width; //how many pixels we send and receive
    int size_of_image = my_image_height * my_image->width;
    int north_neighbor = world_rank + 1;
    int south_neighbor = world_rank - 1;

    pixel *end_of_image;
    pixel *empty_row_north;
    pixel *empty_row_south;

    size_t bytes_to_exchange = num_border_rows * sizeof(pixel) * my_image->width;
    for (unsigned int i = 0; i < options->iterations; i++)
    {
        // If we are only running 1 process we can skip the border exchange
        if (world_sz != 1)
        {
            if (world_rank == 0)
            {
                //points to start of image
                start_of_image = my_image->rawdata;

                //Points to the start of the last row with data in our image
                end_of_image = start_of_image + ((my_image_height - num_border_rows) * my_image->width);

                //Points to where the empty row we need to fill starts
                empty_row_north = start_of_image + size_of_image;

                MPI_Send(
                    end_of_image,                        // Send Buffer
                    nr_of_border_pixels * sizeof(pixel), // Send Count
                    MPI_BYTE,                            // Send Type
                    north_neighbor,                      // Destination
                    0,                                   // TAG
                    MPI_COMM_WORLD                       // Communicator
                );
                MPI_Recv(
                    empty_row_north,                     // Receive Buffer
                    nr_of_border_pixels * sizeof(pixel), // Receive Count
                    MPI_BYTE,                            // Receive Type
                    north_neighbor,                      // Source
                    0,                                   // TAG
                    MPI_COMM_WORLD,                      // Communicator
                    MPI_STATUS_IGNORE                    // Status
                );
            }
            else if (world_rank == last_rank)
            {
                empty_row_south = my_image->rawdata;
                start_of_image = empty_row_south + nr_of_border_pixels;

                // Since the processor before the last will send before it receives we need to receive first
                MPI_Recv(
                    empty_row_south,                     // Receive Buffer
                    nr_of_border_pixels * sizeof(pixel), // Receive Count
                    MPI_BYTE,                            // Receive Type
                    south_neighbor,                      // Source
                    0,                                   // TAG
                    MPI_COMM_WORLD,                      // Communicator
                    MPI_STATUS_IGNORE                    // Status
                );

                // After this we can exhange the last lines of pixels
                MPI_Send(
                    start_of_image,                      // Send Buffer
                    nr_of_border_pixels * sizeof(pixel), // Send Count
                    MPI_BYTE,                            // Send Type
                    south_neighbor,                      // Destination
                    0,                                   // TAG
                    MPI_COMM_WORLD                       // Communicator
                );
            }
            else
            {
                empty_row_south = my_image->rawdata;
                start_of_image = my_image->rawdata + nr_of_border_pixels;

                end_of_image = start_of_image + ((my_image_height - num_border_rows) * my_image->width);
                empty_row_north = start_of_image + size_of_image;

                // Since rank 0 starts sending we need to receive first so we start by exhanging with the south neighbor
                // this is for avoiding deadlocks

                MPI_Recv(
                    empty_row_south,                     // Receive Buffer
                    nr_of_border_pixels * sizeof(pixel), // Receive Count
                    MPI_BYTE,                            // Receive Type
                    south_neighbor,                      // Source
                    0,                                   // TAG
                    MPI_COMM_WORLD,                      // Communicator
                    MPI_STATUS_IGNORE                    // Status
                );

                // After receiving we can send
                MPI_Send(
                    start_of_image,                      // Send Buffer
                    nr_of_border_pixels * sizeof(pixel), // Send Count
                    MPI_BYTE,                            // Send Type
                    south_neighbor,                      // Destination
                    0,                                   // TAG
                    MPI_COMM_WORLD                       // Communicator
                );

                //After we have exhanged with our south neighbor we send our data to our north neighbor
                MPI_Send(
                    end_of_image,                        // Send Buffer
                    nr_of_border_pixels * sizeof(pixel), // Send Count
                    MPI_BYTE,                            // Send Type
                    north_neighbor,                      // Destination
                    0,                                   // TAG
                    MPI_COMM_WORLD                       // Communicator
                );

                // Lastly we receive from our north neighbor which at the end will be the last rank
                MPI_Recv(
                    empty_row_north,                     // Receive Buffer
                    nr_of_border_pixels * sizeof(pixel), // Receive Count
                    MPI_BYTE,                            // Receive Type
                    north_neighbor,                      // Source
                    0,                                   // TAG
                    MPI_COMM_WORLD,                      // Communicator
                    MPI_STATUS_IGNORE                    // Status
                );
            }
        }

        // Apply Kernel
        applyKernel(processImage->data,
                    my_image->data,
                    my_image->width,
                    my_image->height,
                    kernels[options->kernelIndex],
                    kernelDims[options->kernelIndex],
                    kernelFactors[options->kernelIndex]);

        swapImage(&processImage, &my_image);

        // Wait until all ranks have done their part before resuming
        MPI_Barrier(MPI_COMM_WORLD);
    }

    freeImage(processImage);

    // Since we swap images we have to make sure we are pointing to the right image we want to send over the gather call
    if (world_rank == 0)
    {
        start_of_image = my_image->rawdata;
    }
    else
    {
        start_of_image = my_image->rawdata + (num_border_rows * image->width);
    }

    MPI_Gatherv(start_of_image,                // Send Buffer
                bytes_to_transfer[world_rank], // Send Count
                MPI_BYTE,                      // Send Type
                image->rawdata,                // Recv Buffer
                bytes_to_transfer,             // Recv Counts
                displacements,                 // Recv Displacements
                MPI_BYTE,                      // Recv Type
                0,                             // Root
                MPI_COMM_WORLD);               // Communicator

    // Stop timer
    double spentTime = MPI_Wtime();
    printf("Time spent: %.3f seconds\n", (spentTime - starttime));

    if (world_rank == 0)
    {
        //Write the image back to disk
        if (saveImage(image, options->output) < 1)
        {
            fprintf(stderr, "Could not save output to '%s'!\n", options->output);
            freeImage(image);
            abort();
        };
    }

    MPI_Finalize();

graceful_exit:
    options->ret = 0;
error_exit:
    if (options->input != NULL)
        free(options->input);
    if (options->output != NULL)
        free(options->output);
    return options->ret;
};

///////////////////////////////////////////////////////////
//                      Speed-up                         //
//           Running Gaussian with 35 iterations         //
//                     (-k 5 -i 35)                      //
///////////////////////////////////////////////////////////
// 1 process    | time: 64.473 seconds  |     -----      //
// 2 processes  | time: 32.079 seconds  | Speedup: 2.009 //
// 4 processes  | time: 17.609 seconds  | Speedup: 3.661 //
// 8 processes  | time: 20.151 seconds  | Speedup: 3.199 //
///////////////////////////////////////////////////////////
