#include <FeatureLine.hpp>
#include <cmath>
#include <GLFW/glfw3.h>
#include <GL/glew.h>
#include <WindowManager.hpp>
#include <ImageRenderer.hpp>
#include <cstdio>
#include <InputHandler.hpp>
#include <algorithm>



void FL_Point::render()
{

    glUseProgram(SimplePointShader::getInstance()->get());

    glBindVertexArray(VAO);

    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, nullptr);
}

FL_Point::FL_Point(float x, float y)
{
    this->x = x;
    this->y = y;

    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    glGenBuffers(1, &EBO);

    update();
}


float FL_Point::squareDistanceTo(float fromX, float fromY)
{

    const float x_diff = abs(x - fromX);
    const float y_diff = abs(y - fromY);

    return x_diff * x_diff + y_diff * y_diff;

}



void FL_Point::updateParent()
{
    if( parentLine != nullptr )
    {
        parentLine->update();
    }
}

void FL_Point::update()
{
    ManagedWindow *current_window = WindowManager::getPrimary();

    int width, height;

    glfwGetWindowSize(current_window->get(), &width, &height);

    glBindVertexArray(VAO);


    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, nullptr);

    const float delta = 5;

    float vertices[8] = {
        (float) (x - delta),
        (float) (y - delta),
        (float) (x - delta),
        (float) (y + delta),
        (float) (x + delta),
        (float) (y + delta),
        (float) (x + delta),
        (float) (y - delta)
    };

    unsigned int indices[6] = { 0, 1, 2, 2, 3, 0 };

    glBufferData(GL_ARRAY_BUFFER, sizeof(float) * 8, vertices, GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(unsigned int) * 6, indices, GL_STATIC_DRAW);

}

FL_Point::~FL_Point() {
    // TASKS:
    // Delete buffers for this point
    // Delete VAO

    glDeleteBuffers(1, &VBO);
    glDeleteBuffers(1, &EBO);
    glDeleteVertexArrays(1, &VAO);
};


void FeatureLine::_generateBuffers()
{
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    glGenBuffers(1, &EBO);
}

FeatureLine::FeatureLine(float x1, float y1, float x2, float y2) {


    start = new FL_Point{x1, y1};
    end = new FL_Point{x2, y2};

    start->parentLine = this;
    end->parentLine = this;


    _generateBuffers();
    update();
}

FeatureLine::FeatureLine(FL_Point *start, FL_Point *end)
{
    this->start = start;
    this->end = end;

    this->start->parentLine = this;
    this->end->parentLine = this;

    _generateBuffers();

    update();
}

FeatureLine::~FeatureLine()
{
    delete start;
    delete end;
}

void FeatureLine::update()
{
    ManagedWindow *current_window = WindowManager::getPrimary();


    glLineWidth(2);

    int width, height;

    glfwGetWindowSize(current_window->get(), &width, &height);

    glBindVertexArray(VAO);

    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, nullptr);

    float vertices[4] = {
        (float) start->x,
        (float) start->y,
        (float) end->x,
        (float) end->y
    };


    unsigned int indices[2] = { 0, 1 };
    glBufferData(GL_ARRAY_BUFFER, sizeof(float) * 4, vertices, GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(unsigned int) * 2, indices, GL_STATIC_DRAW);


}

void FeatureLine::updateAll()
{
    update();
    start->update();
    end->update();
}

void FeatureLine::render()
{
    // Activate FeatureLine shader
    glUseProgram(FeatureLineShader::getInstance()->get());
    glBindVertexArray(VAO);

    glDrawElements(GL_LINES, 2, GL_UNSIGNED_INT, nullptr);

    start->render();
    end->render();

}

FL_Point *FeatureLine::hit(float x, float y)
{
    float start_dist = start->squareDistanceTo(x, y);
    float end_dist = end->squareDistanceTo(x, y);

    float closest_dist;
    FL_Point *closest;

    if ( end_dist < start_dist )
    {
        closest = end;
        closest_dist = end_dist;
    }
    else
    {
        closest = start;
        closest_dist = start_dist;
    }

    if(closest_dist < FL_POINT_HIT_RADIUS * FL_POINT_HIT_RADIUS)
    {
        closest->is_hit = true;
        return closest;
    }


    return nullptr;
}

FeatureLineManager::FeatureLineManager()
{
    current_orphan = nullptr;
}

FeatureLineManager *FeatureLineManager::instance = nullptr;

FeatureLineManager *FeatureLineManager::getInstance()
{
    if( nullptr == instance )
        instance = new FeatureLineManager{};

    return instance;
}

void FeatureLineManager::setActiveSet(bool first)
{
    activeSet = first;
}

FL_Point *FeatureLineManager::hit(int x, int y)
{

    FL_Point *hit_point;
    for( FL_Pair * pair : line_pairs)
    {
        FeatureLine *&active_point = (activeSet) ? pair->second : pair->first;
        hit_point = active_point->hit(x, y);
        if( nullptr != hit_point )
        {
            return hit_point;
        }
    }

    if( nullptr != current_orphan &&
            current_orphan->squareDistanceTo(x, y) < FL_POINT_HIT_RADIUS * FL_POINT_HIT_RADIUS )
        return current_orphan;


    return nullptr;
}

void FeatureLineManager::add(float x1, float y1, float x2, float y2)
{

    FeatureLine *first = new FeatureLine{x1,y1,x2,y2};
    FeatureLine *second = new FeatureLine{x1,y1,x2,y2};


    FL_Pair *new_pair = new FL_Pair{
        .first=first,
        .second=second
    };

    line_pairs.emplace_back(new_pair);
}

void FeatureLineManager::add(FL_Point *start, FL_Point *end)
{
    FeatureLine *first = new FeatureLine{start, end};
    FeatureLine *second = new FeatureLine{start->x, start->y, end->x, end->y};

    FL_Pair *new_pair = new FL_Pair {
        .first = first,
        .second = second
    };

    line_pairs.emplace_back(new_pair);
}

void FeatureLineManager::remove(FL_Point *point)
{
    if(point == current_orphan)
    {
        delete point;
        current_orphan = nullptr;

        return;
    }

    FeatureLine *line = point->parentLine;

    FL_Pair *pair_containing_line;

    // Search for the FeatureLine in the pairs vector
    for( FL_Pair *pair : line_pairs)
    {
        if( pair->first == line )
        {
            pair_containing_line = pair;
            break;
        } else if ( pair->second == line )
        {
            pair_containing_line = pair;
            break;
        }
    }

    if(pair_containing_line != nullptr)
    {
        line_pairs.erase(std::remove(line_pairs.begin(), line_pairs.end(), pair_containing_line));
    }


}

void FeatureLineManager::render()
{
    for(FL_Pair *pair : line_pairs)
    {

        FeatureLine* activeLine;

        if(activeSet) {
            activeLine = pair->second;
        } else {
            activeLine = pair->first;
        }

        activeLine->render();
    }

    if(current_orphan) current_orphan->render();
}

FeatureLineManager::~FeatureLineManager()
{
    reset();
}

void FeatureLineManager::reset()
{
    for ( FL_Pair *pair : line_pairs )
    {
        delete pair->first;
        delete pair->second;
    }

    line_pairs.clear();
}

void FeatureLineManager::addLinePair(FeatureLine *first, FeatureLine *second)
{
    FL_Pair *pair = new FL_Pair{first, second};

    line_pairs.emplace_back(pair);

}

void FeatureLineManager::addPoint(float x, float y)
{
    FL_Point *new_point = new FL_Point{x,y};

    if( nullptr != current_orphan )
    {
        FeatureLineManager::add(current_orphan, new_point);
        current_orphan = nullptr;
    } else
    {
        current_orphan = new_point;
    }
}

void FeatureLineManager::handleDragging()
{
    if (InputHandler::isClicked(GLFW_MOUSE_BUTTON_LEFT))
    {
        IH_MousePosition pos = InputHandler::getMousePos();

        FL_Point *hit = FeatureLineManager::getInstance()->hit(
                pos.x,
                ImageRenderer::getInstance()->getCurrentImage()->getHeight() - pos.y);

        if(hit) {
            currently_dragging = hit;
        }
    }

    if(InputHandler::isHeld(GLFW_MOUSE_BUTTON_LEFT))
        {

            if(nullptr != currently_dragging)
            {
                IH_MousePosition pos = InputHandler::getMousePos();
                currently_dragging->x = pos.x;
                currently_dragging->y = ImageRenderer::
                    getInstance()->getCurrentImage()->getHeight() - pos.y;

                currently_dragging->update();
                currently_dragging->updateParent();

            }
        }

    if(InputHandler::mouseButtonReleased(GLFW_MOUSE_BUTTON_LEFT))
    {
        currently_dragging = nullptr;
    }
}
